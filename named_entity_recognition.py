# Import modules
import os

# Import custom modules

def main(args):
    dd

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Parsing Method')
    parser.add_argument('--max_len', default=150, type=int)
    parser.add_argument('--save_path', default='./save/', 
                        type=str)
    parser.add_argument('--data_path', default='../joseon_word_embedding/Crawl/crawledResults_NER', 
                        type=str, help='Crawling data path')
    parser.add_argument('--data_split_per', default=0.2, type=float,
                        help='Train / Validation split ratio')
    parser.add_argument('--pad_idx', default=0, type=int, help='Padding index')
    parser.add_argument('--bos_idx', default=1, type=int, help='Start token index')
    parser.add_argument('--eos_idx', default=2, type=int, help='End token index')
    parser.add_argument('--unk_idx', default=3, type=int, help='Unknown token index')
    args = parser.parse_args()

    main(args)