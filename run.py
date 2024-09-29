import argparse
import sys



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
    print(args)