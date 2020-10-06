# Import Modules
import os
import time
import argparse

# Import Custom Modules
from AJD_crawl import AJD_crawl
from AJD_processing import AJD_processing
from JRS_Hanja_crawl import JRS_Hanja_crawl
from JRS_paired_crawl import JRS_paired_crawl

def main(args):

    # Path setting
    if not os.path.exists(args.AJD_main_dir):
        os.mkdir(args.AJD_main_dir)
    if not os.path.exists(args.JRS_main_dir):
        os.mkdir(args.JRS_main_dir)

    # AJD NER & NMT crawling
    AJD_crawl(args)

    # AJD processing
    AJD_processing(args)

    # JRS crawl
    # 1) JRS Hanja crawling
    if args.JRS_Hanja_crawl:
        JRS_Hanja_crawl(args)
    # 2) JRS paired crawling
    if args.JRS_paired_crawl:
        JRS_paired_crawl(args)
    
if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Joseon Dynasty Crawling')
    parser.add_argument('--AJD_main_dir', type=str, default='../data/AJD', 
                        help='Main directory of AJD crawled data')
    parser.add_argument('--JRS_main_dir', type=str, default='../data/JRS', 
                        help='Main directory of JRS crawled data')
    parser.add_argument('--AJD_NER_crawl', type=bool, default=True)
    parser.add_argument('--AJD_translation_crawl', type=bool, default=True)
    parser.add_argument('--JRS_Hanja_crawl', type=bool, default=True)
    parser.add_argument('--JRS_paired_crawl', type=bool, default=True)
    args = parser.parse_args()

    # Main function
    start_time = time.time()
    main(args)
    spend_time = (time.time() - start_time) / 60
    print('Done!; {:2.2%}min spend.'.format(spend_time))