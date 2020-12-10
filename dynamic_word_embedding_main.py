# Import Modules
import time
import argparse

# Import Custom Modules 
from dynamic_word_embedding.preprocessing import preprocessing
from dynamic_word_embedding.training import training

def main(args):
    # Time setting
    total_start_time = time.time()

    # Preprocessing
    if args.preprocessing:
        preprocessing(args)

    # Dynamic word embedding training
    if args.training:
        training(args)

    # Time calculate
    print(f'Done! ; {round((time.time()-total_start_time)/60, 3)}min spend')

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='DWE argparser')
    parser.add_argument('--preprocessing', action='store_true')
    parser.add_argument('--training', action='store_true')
    # Language select
    parser.add_argument('--Hanja_dwe', type=bool, default=True, help='Hanja processing')
    parser.add_argument('--Korean_dwe', type=bool, default=True, help='Korean processing')
    parser.add_argument('--Korean_khaiii', type=bool, default=False)
    # Path setting
    parser.add_argument('--ADJ_data_path', type=str, default='./data', 
                        help='Data path setting')
    parser.add_argument('--save_path', type=str, default='./preprocessing',
                        help='Save path setting')
    # Training setting
    parser.add_argument('--num_epochs', type=int, default=5, help='The number of epoch')
    parser.add_argument('--minibatch_iteration', type=int, default=300, help='Mini-batch iteration')
    parser.add_argument('--pad_idx', default=0, type=int, help='Padding index')
    parser.add_argument('--bos_idx', default=1, type=int, help='Start token index')
    parser.add_argument('--eos_idx', default=2, type=int, help='End token index')
    parser.add_argument('--unk_idx', default=3, type=int, help='Unknown token index')
    parser.add_argument('--vocab_size', default=24000, type=int, help='Korean vocabulary size')
    args = parser.parse_args()

    main(args)