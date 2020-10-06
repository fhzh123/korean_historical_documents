# Import Module
import os
import re
import json
import pickle
import argparse
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm

def AJD_processing(args):
    for folder in sorted(glob(os.path.join(args.AJD_main_dir, '*'))):
        king = folder.split('/')[-1]
        record_list = list()
        print(king)
        for filename in tqdm(glob(os.path.join(folder, '*.txt'))):
            
            # File open
            with open(filename) as f:
                text = f.read()

            # Loop by '\n\n=====\n\n'
            for record in text.split('\n\n=====\n\n')[:-1]:
                korean_content, hanja_content = record.split('\n\n-----\n\n')
                date, title, *korean_content = korean_content.split("\n")
                korean_content = " ".join(korean_content).strip()

                # Blank content processing
                if korean_content == '':
                    continue
                if '。' in korean_content:
                    continue

                # Hanja processing
                # 1) Date processing
                hanja_partition = hanja_content.partition('。')
                if '日' in hanja_partition[0] and len(hanja_partition[0]) <= 10:
                    hanja_content = hanja_partition[2]

                # 2) '/' processing
                if len(hanja_content) > 0:
                    if '/' in hanja_content.split()[0]:
                        hanja_content = ' '.join(hanja_content.split('/')[1:])

                # 3) Symbol processing
                hanja_content = hanja_content.replace('○', ' ')
                hanja_content = hanja_content.replace('。', '。 ')
                hanja_content = hanja_content.replace(',', ', ')
                hanja_content = " ".join(hanja_content.split()).strip()

                # Korean processing
                korean_content = " ".join(korean_content.split()).strip()
                korean_content = " ".join(re.sub(r'\(\w+\)', '', korean_content).split()).strip()
                korean_content = re.sub(r'\d{3}\)', '', korean_content)

                # Data append
                if hanja_content != '':
                    record_list.append({
                        'date': date, 'title': title, 
                        'korean': korean_content, 'hanja': hanja_content
                    })
                    
        # Saving
        with open(os.path.join(args.AJD_main_dir, f'{king}.json'), 'w') as json_file:
            json.dump(record_list, json_file)