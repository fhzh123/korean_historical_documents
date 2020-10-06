# Import Modules
import os
import time
import json
import argparse
import requests
from tqdm import tqdm
from bs4 import BeautifulSoup
from urllib.parse import quote_plus

# Import Custom Modules
from utils import NER_dict_setting, NER_append

def AJD_crawl(args):
    # Main page
    main_url = "http://sillok.history.go.kr/main/main.do"
    main_page = BeautifulSoup(requests.get(main_url).text, 'lxml')

    # Loop by king
    king_areas = main_page.find("div", {"id": "m_cont_list"}).findAll("li")
    for king_area in king_areas:
        area_text = king_area.text
        print(area_text)
        area_dir = f'{args.main_dir}/{area_text}'
        if not os.path.exists(area_dir):
            os.mkdir(area_dir)

        # NER setting
        if args.NER_crawl:
            NER_korean_dict = dict()
            NER_chinese_dict = dict()

        # King page open
        area_id = king_area.find('a')['href'][-6:-3]
        area_page = BeautifulSoup(
            requests.post(
                "http://sillok.history.go.kr/search/inspectionMonthList.do",
                data={'id': area_id}).text, 'lxml')
                
        # Loop by year
        for year in tqdm(area_page.find('ul', {'class':
                                        'king_year2 clear2'}).findAll('ul')):

            # Loop by month
            for month in year.findAll('a'):
                month_text = month.text
                id_ = month['href'].split("'")[1]
                url = f"http://sillok.history.go.kr/search/inspectionDayList.do?id={id_}"
                month_page = BeautifulSoup(requests.get(url).text, 'lxml')
                time.sleep(0.2)
                month_page = month_page.find("dl", {'class': 'ins_list_main'})
                book_section_text = month_page.find('dt').text.strip()

                fwrite = open(f"{area_dir}/{book_section_text}.txt", 'w', encoding='utf-8')

                # Loop by event
                for event in month_page.findAll('a'):
                    event_id = event['href'].split("'")[1]
                    event_id = quote_plus(event_id)
                    event_url = f"http://sillok.history.go.kr/id/{event_id}"
                    event_page = BeautifulSoup(requests.get(event_url).text, 'lxml')

                    event_time = event_page.find(
                        'span', {'class': 'tit_loc'}).text.strip()
                    event_time = ' '.join(event_time.split())
                    event_text = event_page.find(
                        'h3', {'class': 'search_tit ins_view_tit'}).text.strip()
                    fwrite.write(f"{event_time}\n")
                    fwrite.write(f"{event_text}\n\n\n")

                    event_korean = event_page.find(
                        'div', {'class': 'ins_view_in ins_left_in'}).find(
                            'div', {'class': 'ins_view_pd'})
                    event_chinese = event_page.find(
                        'div', {'class': 'ins_view_in ins_right_in'}).find(
                            'div', {'class': 'ins_view_pd'})

                    # NER dict setting (in utils.py)
                    NER_korean_dict, NER_chinese_dict = NER_dict_setting(
                        NER_korean_dict, NER_chinese_dict, event_id
                    )

                    for paragraph_korean in event_korean.findAll(
                            "p", {'class': 'paragraph'}):
                        # NER processing
                        if args.NER_crawl:
                            for idx in paragraph_korean.findAll("span"):
                                attr_class_list = idx.get_attribute_list('class')
                                # NER added (in utils.py)
                                NER_korean_dict = NER_append(NER_korean_dict, 
                                                             event_id, attr_class_list)
                        # NMT processing
                        if args.translation_crawl:
                            paragraph_korean = paragraph_korean.text
                            if paragraph_korean.startswith("○"):
                                break

                            fwrite.write(' '.join(paragraph_korean.split()))
                            fwrite.write('\n')

                    if args.translation_crawl:
                        fwrite.write('\n-----\n\n')

                    for paragraph_chinese in event_chinese.findAll(
                            'p', {'class': 'paragraph'}):
                        # NER Processing
                        if args.NER_crawl:
                            for idx in paragraph_chinese.findAll("span"):
                                attr_class_list = idx.get_attribute_list('class')
                                # NER added (in utils.py)
                                NER_chinese_dict = NER_append(NER_chinese_dict, 
                                                             event_id, attr_class_list)
                        if args.translation_crawl:
                            fwrite.write(' '.join(paragraph_chinese.text.split()))
                            fwrite.write('\n')

                    if args.translation_crawl:
                        fwrite.write('\n=====\n\n')

                fwrite.close()

        # Save NER processing
        with open(f'{area_dir}/NER_korean.json', 'w') as fp:
            json.dump(NER_korean_dict, fp)

        with open(f'{area_dir}/NER_chinese.json', 'w') as fp:
            json.dump(NER_chinese_dict, fp)

    # Prevent DDos defense
    time.sleep(0.5)