# Import modules
import os
import time
import argparse

# Import custom modules
from neural_machine_translation.preprocessing import preprocessing
from neural_machine_translation.training import training

def main(args):
    # Time setting
    total_start_time = time.time()

    # NER preprocessing
    if args.preprocessing:
        preprocessing(args)

    # NER main
    training(args)

    # Time calculate
    print(f'Done! ; {round((time.time()-total_start_time)/60, 3)}min spend')

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='NMT argparser')
    # Path setting
    parser.add_argument('--save_path', default='./preprocessing', 
                        type=str)
    parser.add_argument('--ADJ_data_path', default='./data', 
                        type=str, help='Crawling data path')
    # Baseline checking
    parser.add_argument('--src_baseline', action='store_true',
                        help='If not store, then training from Dynamic Word Embedding')
    parser.add_argument('--trg_baseline', action='store_true',
                        help='If not store, then training from Dynamic Word Embedding')
    # Preprocessing setting
    parser.add_argument('--preprocessing', action='store_true')
    parser.add_argument('--dataset', default='normal_data', choices=['normal_data', 'total_data'],
                        type=str, help='Data selecting; normal_data is preprocessed pickle file & total_data is sjw total dataset')
    parser.add_argument('--data_split_per', default=0.2, type=float,
                        help='Train / Validation split ratio')
    parser.add_argument('--hj_vocab_size', default=32000, type=int, help='Hanja vocabulary size; Default is 32000')
    parser.add_argument('--kr_vocab_size', default=32000, type=int, help='Korean vocabulary size; Default is 32000')
    parser.add_argument('--pad_idx', default=0, type=int, help='pad index')
    parser.add_argument('--bos_idx', default=1, type=int, help='index of bos token')
    parser.add_argument('--eos_idx', default=2, type=int, help='index of eos token')
    parser.add_argument('--unk_idx', default=3, type=int, help='index of unk token')
    # Model setting
    parser.add_argument('--model_setting', default='transformer', choices=['transformer', 'rnn'],
                        type=str, help='Model Selection; transformer vs rnn')
    parser.add_argument('--resume', action='store_true',
                        help='If not store, then training from scratch')
    parser.add_argument('--d_model', type=int, default=512, help='Hidden State Vector Dimension; Default is 512')
    parser.add_argument('--d_embedding', type=int, default=256, help='Embedding Vector Dimension; Default is 256')
    parser.add_argument('--n_head', type=int, default=8, help='Multihead Count; Default is 256')
    parser.add_argument('--dim_feedforward', type=int, default=512, help='Embedding Vector Dimension; Default is 512')
    parser.add_argument('--num_encoder_layer', default=8, type=int, help='number of encoder layer')
    parser.add_argument('--num_decoder_layer', default=8, type=int, help='number of decoder layer')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout Ratio; Default is 0.5')
    # Training setting
    parser.add_argument('--min_len', type=int, default=4, help='Minimum Length of Sentences; Default is 4')
    parser.add_argument('--max_len', type=int, default=300, help='Max Length of Source Sentence; Default is 150')
    parser.add_argument('--num_epoch', type=int, default=10, help='Epoch count; Default is 10')
    parser.add_argument('--batch_size', type=int, default=48, help='Batch size; Default is 48')
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate; Default is 5e-5')
    parser.add_argument('--lr_decay', type=float, default=0.5, help='Learning rate decay; Default is 0.5')
    parser.add_argument('--lr_decay_step', type=int, default=2, help='Learning rate decay step; Default is 5')
    parser.add_argument('--grad_clip', type=int, default=5, help='Set gradient clipping; Default is 5')
    parser.add_argument('--w_decay', type=float, default=1e-6, help='Weight decay; Default is 1e-6')
    parser.add_argument('--print_freq', type=int, default=300, help='Print train loss frequency; Default is 100')
    args = parser.parse_args()

    main(args)