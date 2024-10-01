from absl import flags
import collections
import logging
import multiprocessing
import os
import pickle
import random
import sys
import time

import tensorflow as tf

from util import *
from vocab import *

random_seed = 12345
short_seq_prob = 0  # Probability of creating sequences which are shorter than the maximum length
logger = tf.get_logger()
logger.setLevel(logging.INFO)

FLAGS = flags.FLAGS
flags.DEFINE_string("signature", 'default', "signature_name")
flags.DEFINE_integer(
    "pool_size", 10,
    "multiprocesses pool size.")
flags.DEFINE_integer(
    "max_seq_length", 200,
    "max sequence length.")
flags.DEFINE_integer(
    "max_predictions_per_seq", 20,
    "max_predictions_per_seq.")
flags.DEFINE_float(
    "masked_lm_prob", 0.15,
    "Masked LM probability.")
flags.DEFINE_float(
    "mask_prob", 1.0,
    "mask probabaility")
flags.DEFINE_integer(
    "dupe_factor", 10,
    "Number of times to duplicate the input data (with different masks).")
flags.DEFINE_float("prop_sliding_window", 0.1, "sliding window step size.")
flags.DEFINE_string(
    "data_dir", './data/',
    "data dir.")
flags.DEFINE_string(
    "dataset_name", 'ml-1m',
    "dataset name.")

MaskedLmInstance = collections.namedtuple("MaskedLmInstance",
                                          ["index", "label"])

class TrainingInstance(object):
    """A single training instance (sentence pair)."""

    def __init__(self, info, tokens, masked_lm_positions, masked_lm_labels):
        self.info = info  # info = [user]
        self.tokens = tokens
        self.masked_lm_positions = masked_lm_positions
        self.masked_lm_labels = masked_lm_labels

    def __str__(self):
        s = ""
        s += f"info: {' '.join(map(str, self.info))}\n"
        s += f"tokens: {' '.join(map(str, self.tokens))}\n"
        s += f"masked_lm_positions: {' '.join(map(str, self.masked_lm_positions))}\n"
        s += f"masked_lm_labels: {' '.join(map(str, self.masked_lm_labels))}\n"
        s += f"\n"
        return s

    def __repr__(self):
        return self.__str__()
    
def create_int_feature(values):
    feature = tf.train.Feature(
        int64_list=tf.train.Int64List(value=list(values)))
    return feature

def create_float_feature(values):
    feature = tf.train.Feature(
        float_list=tf.train.FloatList(value=list(values)))
    return feature

def write_instance_to_example_files(instances, max_seq_length,
                                    max_predictions_per_seq, vocab,
                                    output_files):
    """Create TF example files from `TrainingInstance`s."""
    writers = []
    for output_file in output_files:
        writers.append(tf.io.TFRecordWriter(output_file))
    
    writer_index = 0
    total_written = 0

    for inst_index, instance in enumerate(instances):
        try:
            input_ids = vocab.convert_tokens_to_ids(instance.tokens)
        except Exception as e:
            logger.info(f"Error ({e}) converting tokens to IDs for instance {instance}")
            continue

        input_mask = [1] * len(input_ids)
        assert len(input_ids) <= max_seq_length
        input_ids += [0] * (max_seq_length - len(input_ids))
        input_mask += [0] * (max_seq_length - len(input_mask))
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        
        masked_lm_positions = list(instance.masked_lm_positions)
        masked_lm_ids = vocab.convert_tokens_to_ids(instance.masked_lm_labels)
        masked_lm_weights = [1.0] * len(masked_lm_ids)
        masked_lm_positions += [0] * (max_predictions_per_seq - len(masked_lm_positions))
        masked_lm_ids += [0] * (max_predictions_per_seq - len(masked_lm_ids))
        masked_lm_weights += [0.0] * (max_predictions_per_seq - len(masked_lm_weights))

        features = {
            "info": create_int_feature(instance.info),
            "input_ids": create_int_feature(input_ids),
            "input_mask": create_int_feature(input_mask),
            "masked_lm_positions": create_int_feature(masked_lm_positions),
            "masked_lm_ids": create_int_feature(masked_lm_ids),
            "masked_lm_weights": create_float_feature(masked_lm_weights)
        }
        tf_example = tf.train.Example(
            features=tf.train.Features(feature=features))
        
        writers[writer_index].write(tf_example.SerializeToString())
        writer_index = (writer_index + 1) % len(writers)
        total_written += 1

        if inst_index < 20:
            logger.info("*** Example ***")
            logger.info(f"tokens: {' '.join(map(str, instance.tokens))}")

            for feature_name, feature in features.items():
                values = []
                if feature.int64_list.value:
                    values = feature.int64_list.value
                elif feature.float_list.value:
                    values = feature.float_list.value
                logger.info(f"{feature_name}: {' '.join(map(str, values))}")

    for writer in writers:
        writer.close()

    logger.info(f"Wrote {total_written} total instances")

def create_masked_lm_predictions_force_last(tokens):
    """Creates the predictions for the masked LM objective."""

    last_index = -1
    for (i, token) in enumerate(tokens):
        if token == "[CLS]" or token == "[PAD]" or token == '[NO_USE]':
            continue
        last_index = i

    assert last_index > 0

    output_tokens = list(tokens)
    output_tokens[last_index] = "[MASK]"

    masked_lm_positions = [last_index]
    masked_lm_labels = [tokens[last_index]]

    return (output_tokens, masked_lm_positions, masked_lm_labels)

def create_instances_from_document_test(all_documents, user, max_seq_length):
    """Creates `TrainingInstance`s for a single document."""
    document = all_documents[user]
    max_num_tokens = max_seq_length
    
    assert len(document) == 1 and len(document[0]) <= max_num_tokens
    
    tokens = document[0]
    assert len(tokens) >= 1

    (tokens, masked_lm_positions,
     masked_lm_labels) = create_masked_lm_predictions_force_last(tokens)

    info = [int(user.split("_")[1])]
    instance = TrainingInstance(
        info=info,
        tokens=tokens,
        masked_lm_positions=masked_lm_positions,
        masked_lm_labels=masked_lm_labels)

    return [instance]

def create_masked_lm_predictions(tokens, masked_lm_prob,
                                 max_predictions_per_seq, vocab_words, rng,
                                 mask_prob):
    """Creates the predictions for the masked LM objective."""

    cand_indexes = []
    for i, token in enumerate(tokens):
        if token not in vocab_words:
            continue
        cand_indexes.append(i)

    rng.shuffle(cand_indexes)

    output_tokens = list(tokens)

    num_to_predict = min(max_predictions_per_seq,
                         max(1, int(round(len(tokens) * masked_lm_prob))))

    masked_lms = []
    covered_indexes = set()
    for index in cand_indexes:
        if len(masked_lms) >= num_to_predict:
            break
        if index in covered_indexes:
            continue
        covered_indexes.add(index)

        masked_token = None
        # 80% of the time, replace with [MASK]
        if rng.random() < mask_prob:
            masked_token = "[MASK]"
        else:
            # 10% of the time, keep original
            if rng.random() < 0.5:
                masked_token = tokens[index]
            # 10% of the time, replace with random word
            else:
                # masked_token = vocab_words[rng.randint(0, len(vocab_words) - 1)]
                masked_token = rng.choice(vocab_words)  

        output_tokens[index] = masked_token

        masked_lms.append(MaskedLmInstance(index=index, label=tokens[index]))

    masked_lms = sorted(masked_lms, key=lambda x: x.index)

    masked_lm_positions = []
    masked_lm_labels = []
    for p in masked_lms:
        masked_lm_positions.append(p.index)
        masked_lm_labels.append(p.label)

    return (output_tokens, masked_lm_positions, masked_lm_labels)

def create_instances_from_document_train(
        all_documents, user, max_seq_length, short_seq_prob, masked_lm_prob,
        max_predictions_per_seq, vocab, rng, mask_prob):
    """Creates `TrainingInstance`s for a single document."""
    document = all_documents[user]

    max_num_tokens = max_seq_length

    instances = []
    info = [int(user.split("_")[1])]
    vocab_items = vocab.get_items()

    for tokens in document:
        assert len(tokens) >= 1 and len(tokens) <= max_num_tokens
        
        (tokens, masked_lm_positions,
         masked_lm_labels) = create_masked_lm_predictions(
             tokens, masked_lm_prob, max_predictions_per_seq,
             vocab_items, rng, mask_prob)
        instance = TrainingInstance(
            info=info,
            tokens=tokens,
            masked_lm_positions=masked_lm_positions,
            masked_lm_labels=masked_lm_labels)
        instances.append(instance)

    return instances

def create_instances_threading(all_documents, user, max_seq_length, short_seq_prob,
                               masked_lm_prob, max_predictions_per_seq, vocab, rng,
                               mask_prob, step):
    cnt = 0
    start_time = time.time()
    instances = []
    for user in all_documents:
        cnt += 1
        if cnt % 1000 == 0:
            print(f"step: {step}, name: {multiprocessing.current_process().name}, step: {cnt}, time: {time.time()-start_time}")
            start_time = time.time()
        instances.extend(create_instances_from_document_train(
            all_documents, user, max_seq_length, short_seq_prob,
            masked_lm_prob, max_predictions_per_seq, vocab, rng,
            mask_prob))
        
    return instances

def mask_last(
        all_documents, user, max_seq_length, short_seq_prob, masked_lm_prob,
        max_predictions_per_seq, vocab, rng):
    """Creates `TrainingInstance`s for a single document."""
    document = all_documents[user]
    max_num_tokens = max_seq_length
    
    instances = []
    info = [int(user.split("_")[1])]
    vocab_items = vocab.get_items()

    for tokens in document:
        assert len(tokens) >= 1 and len(tokens) <= max_num_tokens
        
        (tokens, masked_lm_positions,
         masked_lm_labels) = create_masked_lm_predictions_force_last(tokens)
        instance = TrainingInstance(
            info=info,
            tokens=tokens,
            masked_lm_positions=masked_lm_positions,
            masked_lm_labels=masked_lm_labels)
        instances.append(instance)

    return instances

def create_training_instances(all_documents_raw,
                              max_seq_length,
                              dupe_factor,
                              short_seq_prob,
                              masked_lm_prob,
                              max_predictions_per_seq,
                              rng,
                              vocab,
                              mask_prob,
                              prop_sliding_window,
                              pool_size,
                              force_last=False):
    """Create `TrainingInstance`s from raw text."""
    all_documents = {}

    if force_last:
        max_num_tokens = max_seq_length
        for user, item_seq in all_documents_raw.items():
            if len(item_seq) == 0:
                logger.info("got empty seq:" + user)
                continue
            all_documents[user] = [item_seq[-max_num_tokens:]]
    else:
        max_num_tokens = max_seq_length  # we need two sentence

        sliding_step = (int)(
            prop_sliding_window *
            max_num_tokens) if prop_sliding_window != -1.0 else max_num_tokens
        for user, item_seq in all_documents_raw.items():
            if len(item_seq) == 0:
                logger.info("got empty seq:" + user)
                continue

            #todo: add slide
            if len(item_seq) <= max_num_tokens:
                all_documents[user] = [item_seq]
            else:
                beg_idx = list(range(len(item_seq)-max_num_tokens, 0, -sliding_step))
                beg_idx.append(0)
                all_documents[user] = [item_seq[i:i + max_num_tokens] for i in beg_idx[::-1]]

    instances = []
    if force_last:
        for user in all_documents:
            instances.extend(
                create_instances_from_document_test(
                    all_documents, user, max_seq_length))
        logger.info("num of instance:{}".format(len(instances)))
    else:
        start_time = time.time()
        pool = multiprocessing.Pool(processes=pool_size)
        logger.info("document num: {}".format(len(all_documents)))

        def log_result(result):
            logger.info("callback function result type: {}, size: {} ".format(type(result), len(result)))
            instances.extend(result)

        for step in range(dupe_factor):
            pool.apply_async(
                create_instances_threading, args=(
                    all_documents, user, max_seq_length, short_seq_prob,
                    masked_lm_prob, max_predictions_per_seq, vocab, random.Random(random.randint(1,10000)),
                    mask_prob, step), callback=log_result)
        pool.close()
        pool.join()
        
        for user in all_documents:
            instances.extend(
                mask_last(
                    all_documents, user, max_seq_length, short_seq_prob,
                    masked_lm_prob, max_predictions_per_seq, vocab, rng))

        logger.info("num of instance:{}; time:{}".format(len(instances), time.time() - start_time))
    rng.shuffle(instances)
    return instances

def gen_samples(data,
                output_filename,
                rng,
                vocab,
                max_seq_length,
                dupe_factor,
                short_seq_prob,
                mask_prob,
                masked_lm_prob,
                max_predictions_per_seq,
                prop_sliding_window,
                pool_size,
                force_last=False):
    # create train
    instances = create_training_instances(
        data, max_seq_length, dupe_factor, short_seq_prob, masked_lm_prob,
        max_predictions_per_seq, rng, vocab, mask_prob, prop_sliding_window,
        pool_size, force_last)

    logger.info("*** Writing to output files ***")
    logger.info("  %s", output_filename)

    write_instance_to_example_files(instances, max_seq_length,
                                    max_predictions_per_seq, vocab,
                                    [output_filename])

def main():
    # Parse arguments.
    FLAGS(sys.argv)
    max_seq_length = FLAGS.max_seq_length
    max_predictions_per_seq = FLAGS.max_predictions_per_seq
    masked_lm_prob = FLAGS.masked_lm_prob
    mask_prob = FLAGS.mask_prob
    dupe_factor = FLAGS.dupe_factor
    prop_sliding_window = FLAGS.prop_sliding_window
    pool_size = FLAGS.pool_size
    output_dir = FLAGS.data_dir
    dataset_name = FLAGS.dataset_name
    version_id = FLAGS.signature
    logger.info(f"Version ID: {version_id}")

    # Check output_dir.
    if not os.path.isdir(output_dir):
        logger.error(output_dir + ' is not exist')
        logger.error(os.getcwd())
        exit(1)

    # Parse and partition dataset.
    dataset = data_partition(os.path.join(output_dir, f"{dataset_name}.txt"))
    user_train, user_valid, user_test, usernum, itemnum = dataset
    
    # Calculate dataset stats.
    cc = 0
    max_len = 0
    min_len = 100000
    for u in user_train:
        cc += len(user_train[u])
        max_len = max(len(user_train[u]), max_len)
        min_len = min(len(user_train[u]), min_len)
    logger.info(f'average sequence length: {cc / len(user_train):.2f}')
    logger.info(f'max:{max_len}, min:{min_len}')
    logger.info(f'len_train:{len(user_train)}, len_valid:{len(user_valid)}, len_test:{len(user_test)}, usernum:{usernum}, itemnum:{itemnum}')

    # Append validation part into train part.
    for u in user_train:
        if u in user_valid:
            user_train[u].extend(user_valid[u])

    # Create tokenized dataset (not masked yet).
    user_train_data = {
        'user_' + str(k): ['item_' + str(item) for item in v]
        for k, v in user_train.items() if len(v) > 0
    }
    user_test_data = {
        'user_' + str(u):
            ['item_' + str(item) for item in (user_train[u] + user_test[u])]
        for u in user_train if len(user_train[u]) > 0 and len(user_test[u]) > 0
    }
    rng = random.Random(random_seed)
    vocab = FreqVocab(user_test_data)
    user_test_data_output = {
        k: [vocab.convert_tokens_to_ids(v)]
        for k, v in user_test_data.items()
    }

    logger.info('begin to generate train')
    output_filename = output_dir + dataset_name + version_id + '.train.tfrecord'
    gen_samples(
        user_train_data,
        output_filename,
        rng,
        vocab,
        max_seq_length,
        dupe_factor,
        short_seq_prob,
        mask_prob,
        masked_lm_prob,
        max_predictions_per_seq,
        prop_sliding_window,
        pool_size,
        force_last=False)
    logger.info(f'train:{output_filename}')

    logger.info('begin to generate test')
    output_filename = output_dir + dataset_name + version_id + '.test.tfrecord'
    gen_samples(
        user_test_data,
        output_filename,
        rng,
        vocab,
        max_seq_length,
        dupe_factor,
        short_seq_prob,
        mask_prob,
        masked_lm_prob,
        max_predictions_per_seq,
        -1.0,
        pool_size,
        force_last=True)
    logger.info(f'test:{output_filename}')

    logger.info(f'vocab_size:{vocab.get_vocab_size()}, user_size:{vocab.get_user_count()}, item_size:{vocab.get_item_count()}, item_with_other_size:{vocab.get_item_count() + vocab.get_special_token_count()}')
    vocab_file_name = output_dir + dataset_name + version_id + '.vocab'
    logger.info('vocab pickle file: ' + vocab_file_name)
    with open(vocab_file_name, 'wb') as output_file:
        pickle.dump(vocab, output_file, protocol=2)

    his_file_name = output_dir + dataset_name + version_id + '.his'
    logger.info('test data pickle file: ' + his_file_name)
    with open(his_file_name, 'wb') as output_file:
        pickle.dump(user_test_data_output, output_file, protocol=2)
    logger.info('done.')


if __name__ == "__main__":
    main()