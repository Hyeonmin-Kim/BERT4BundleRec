import argparse
import logging
import os
import pickle

import tensorflow as tf

import model


logger = tf.get_logger()
logger.setLevel(logging.DEBUG)


def main(args):

    # Manipulate arguments.
    args.checkpointDir += args.signature
    assert args.do_train or args.do_eval, "At least one of `do_train` or `do_eval` must be True."
    bert_config = model.BertConfig.from_json_file(args.bert_config_file)

    # Collect input file paths.
    os.makedirs(os.path.join(os.getcwd(), args.checkpointDir), exist_ok=True)
    train_input_filepaths = args.train_input_file.split(',')
    test_input_filepaths = args.test_input_file.split(',') if args.test_input_file else train_input_filepaths
    logger.info("*** train Input Files ***")
    for input_file in train_input_filepaths:
        logger.info(input_file)
    logger.info("*** test Input Files ***")
    for input_file in test_input_filepaths:
        logger.info(input_file)

    # Load vocab file.
    with open(args.vocab_filename, 'rb') as vocab_file:
        vocab = pickle.load(vocab_file)
    item_size = len(vocab.counter)

    # Test BertModel
    bert_model = model.BertModel(bert_config, use_one_hot_embeddings=True)
    test_input = {
        'input_ids': tf.constant([[31, 51, 99], [15, 5, 0]], dtype=tf.int32),
        'input_mask': tf.constant([[1, 1, 1], [1, 1, 0]], dtype=tf.int32),
        'token_type_ids': tf.constant([[0, 0, 1], [0, 1, 0]], dtype=tf.int32),
    }
    res = bert_model(test_input)
    print(res.shape)


def get_argparser():
    parser = argparse.ArgumentParser(description='BERT4Rec Training and Evaluation')
    parser.add_argument('--bert_config_file', required=True, help='The config json file corresponding to the pre-trained BERT model. This specifies the model architecture.')
    parser.add_argument('--train_input_file', required=True, help='Input TF example files (can be a glob or comma separated).')
    parser.add_argument('--test_input_file', required=True, help='Input TF example files (can be a glob or comma separated).')
    parser.add_argument('--checkpointDir', required=True, help='The output directory where the model checkpoints will be written.')
    parser.add_argument('--vocab_filename', required=True, help='vocab filename')
    parser.add_argument('--user_history_filename', required=True, help='user history filename')

    parser.add_argument('--signature', default='default', help='signature_name')
    parser.add_argument('--max_seq_length', default=128, help='The maximum total input sequence length after WordPiece tokenization. Sequences longer than this will be truncated, and sequences shorter than this will be padded. Must match data generation.')
    parser.add_argument('--init_checkpoint', help='Initial checkpoint (usually from a pre-trained BERT model).')
    parser.add_argument('--max_predictions_per_seq', default=20, help='Maximum number of masked LM predictions per sequence. Must match data generation.')
    parser.add_argument('--do_train', default=False, help='Whether to run training.')
    parser.add_argument('--do_eval', default=False, help='Whether to run eval on the dev set.')
    parser.add_argument('--batch_size', default=32, help='Total batch size for training.')
    parser.add_argument('--learning_rate', default=5e-5, help='The initial learning rate for Adam.')
    parser.add_argument('--num_train_steps', default=100000, help='Number of training steps.')
    parser.add_argument('--num_warmup_steps', default=10000, help='Number of warmup steps.')
    parser.add_argument('--save_checkpoints_steps', default=1000, help='How often to save the model checkpoint.')
    parser.add_argument('--iterations_per_loop', default=1000, help='How many steps to make in each estimator call.')
    parser.add_argument('--max_eval_steps', default=1000, help='Maximum number of eval steps.')
    parser.add_argument('--use_tpu', default=False, help='Whether to use TPU or GPU/CPU.')
    parser.add_argument('--tpu_name', help='The Cloud TPU to use for training. This should be either the name used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 url.')
    parser.add_argument('--tpu_zone', help='[Optional] GCE zone where the Cloud TPU is located in. If not specified, we will attempt to automatically detect the GCE project from metadata.')
    parser.add_argument('--gcp_project', help='[Optional] Project name for the Cloud TPU-enabled project. If not specified, we will attempt to automatically detect the GCE project from metadata.')
    parser.add_argument('--master', help='[Optional] TensorFlow master URL.')
    parser.add_argument('--num_tpu_cores', default=8, help='Only used if `use_tpu` is True. Total number of TPU cores to use.')
    parser.add_argument('--use_pop_random', default=True, help='use pop random negative samples')
    
    return parser



if __name__ == "__main__":
    parser = get_argparser()
    args = parser.parse_args()
    main(args)