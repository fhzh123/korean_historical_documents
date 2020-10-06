# Import Modules
import os
import json
import time
import requests
import pandas as pd
from bs4 import BeautifulSoup
from itertools import product
from urllib.parse import quote_plus

def JRS_paired_crawl(args):

    # Making main directory
    main_dir = os.path.join(args.JRS_main_dir, 'paired')
    king_list = [('P0', range(1, 28),'InJo'), ('U0', range(1, 7),'YoungJo'), ('Z0', range(1, 44),'GoJong'), ('ZA', range(1, 5),'SoonJong')]

    # URL setting
    url = 'http://db.itkc.or.kr/dir/node?grpId=&itemId=ST&gubun=book&depth=5&cate1=&cate2=&dataId=ITKC_ST_{}_A{:02d}_{}_{:02d}A_{:04d}0'
    
    # Error prepare setting
    except_day_list = list()
    except_month_list = list()
    except_year_list = list()
    except_king_list = list()

    # Loop by king & File setting
    for king, years, king_area in king_list:
        area_dir = f'{main_dir}/{king_area}'
        if not os.path.exists(area_dir):
            os.mkdir(area_dir)

        # Loop by year
        for year in years:
            # Loop by month (A, B is for leap year)
            for month, month_version in product(range(1, 13), ('A', 'B')):
                month_string = f'{month:02d}{month_version}'
                previous_url, previous_hanja, previous_korean, date, res_compare = str(), str(), str(), str(), str()
                articles = list()

                # Loop by day
                for day in range(1, 32):
                    print(f'{king_area}: {year}year {month}month {month_version} crawling...')
                    # Try / Except => Unknown error occur sometime
                    try:
                        event_id = 1
                        while True:
                            # URL setting
                            event_id += 1
                            url_korean = url.format(king, year, month_string, day, event_id)
                            main_page = BeautifulSoup(requests.get(url_korean).text, 'lxml')
                            event = main_page.find('div', {'class': 'fs16 tac'})
                            res = requests.get(url_korean)
                            
                            # Error & Blank page skip
                            if res.status_code != 200:
                                break
                            if event != None:
                                continue

                            # Korean text
                            korean_text = ""
                            for book in main_page.findAll("div", {"data-xsl-gubun": "최종정보"}):
                                for korean in book.findAll("div", {"style": "text-align:left"}):
                                    korean_text += korean.text

                            # Hanja text
                            id_hanja = main_page.find("div", {'class': 'option_btn_txt fr mr10'})
                            if id_hanja.find('a') == None:
                                continue
                            id_hanja = id_hanja.find('a')['href']
                            url_hanja = f"http://db.itkc.or.kr{id_hanja}"

                            res_compare = requests.get(url_hanja).url
                            if previous_url != "":
                                if previous_url == res_compare:
                                    previous_korean += " " + korean_text
                                else:
                                    hanja = BeautifulSoup(requests.get(url_hanja).text, 'lxml')
                                    if hanja.find('div', {'class': 'p_20'}) == None:
                                        continue
                                    hanja_event = hanja.find('div', {'class': 'p_20'})
                                    articles.append({'hanja': previous_hanja,
                                                    'korean': previous_korean,
                                                    'date': date
                                                    })
                                    date = hanja.find('span', {'class': 'tit_loc'}).text
                                    previous_hanja = " ".join(hanja_event.text.split())
                                    date = " ".join(date[date.find(")"):][3:].split()[:4])
                                    previous_korean = " ".join(korean_text.split())

                            else:
                                hanja = BeautifulSoup(requests.get(url_hanja).text, 'lxml')
                                if hanja.find('div', {'class': 'p_20'}) == None:
                                    continue
                                hanja_event = hanja.find('div', {'class': 'p_20'})
                                articles.append({'hanja': previous_hanja,
                                                'korean': previous_korean,
                                                'date': date
                                                })
                                date = hanja.find('span', {'class': 'tit_loc'}).text
                                date = " ".join(date[date.find(")"):][3:].split()[:4])
                                previous_hanja = " ".join(hanja_event.text[1:].split())
                                previous_korean = " ".join(korean_text.split())
                            previous_url = res_compare
                            time.sleep(0.3)

                    # For unknown error
                    except:
                        count += 1
                        print('Error occur:')
                        print(f'{king} | year: {year} | month: {month} | day: {day}')
                        except_day_list.append(day)
                        except_month_list.append(month)
                        except_year_list.append(year)
                        except_king_list.append(king)

                # Saving
                fwrite = open(f"{area_dir}/{king_area}_Year{year}_Month{month}_{month_version}.json", 'w', encoding='utf-8')
                json.dump(articles[1:], fwrite, indent='\t')
                fwrit.close()

    # Error saving
    if count >= 1:
        pd.DataFrame({
            'day': except_day_list,
            'month': except_month_list,
            'year': except_year_list,
            'king': except_king_list
        }).to_csv(os.path.join(main_dir, 'except_list.csv'), index=False)
        print(f'Total {count} error occur.')