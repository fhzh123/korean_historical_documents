# Import Module
import os
import time
import pickle
import requests
from bs4 import BeautifulSoup
from urllib.parse import quote_plus

def JRS_Hanja_crawl(args):

    # Main crawling URL
    main_url = "http://sjw.history.go.kr/main.do"
    main_page = BeautifulSoup(requests.get(main_url).text, 'lxml')

    # Making main directory
    main_dir = os.path.join(args.JRS_main_dir, 'Hanja')

    # Making king list
    king_areas_ = main_page.find("div", {"class": "m_cont_top"}).findAll("li")

    # Loop by king
    for king_area_ in king_areas_[1:9]:
        area_text = king_area_.text

        # Path setting
        area_dir = f'{main_dir}/{area_text}'
        if not os.path.exists(area_dir):
            os.mkdir(area_dir)

        area_id = king_area_.find('a')['href'][-14:-9]
        url = f"http://sjw.history.go.kr/search/inspectionMonthList.do?TreeID={area_id}"

        # Try / Except => Unknown error occur sometime
        try:
            area_page = BeautifulSoup(requests.post(url).text, 'lxml')
        except:
            print(f"{king_area_} Error | Restarting...")
            time.sleep(100)
            area_page = BeautifulSoup(requests.post(url).text, 'lxml')

        # Loop by year
        for year in area_page.find('ul', {'class': 'king_year2 clear2'}).findAll('ul'):
            # Loop by month
            for month in year.findAll('a'):
                month_text = month.text

                # URL setting
                articles = list()
                id_ = month['href'].split("'")[1]
                url = f"http://sjw.history.go.kr/search/inspectionDayList.do?TreeID={id_}"

                # Try / Except => Unknown error occur sometime
                try:
                    month_page = BeautifulSoup(requests.get(url).text, 'lxml')
                except:
                    print(f"{king_area_}_{month.text} Error | Restarting...")
                    time.sleep(100)
                    month_page = BeautifulSoup(requests.get(url).text, 'lxml')

                book_month_page = month_page.find("div", {'class': 'view_tit'})
                book_section_text = book_month_page.find('span').text.strip()
                book_section_text = book_section_text.split("월")[0] + "월"
                print(book_section_text)

                # Pickle file write
                with open(f"{area_dir}/{book_section_text}.pkl", 'wb') as fwrite:

                    month_page = month_page.find('span', {'class': 'day_list'})

                    # Loop by day
                    for event in month_page.findAll('a'):
                        # First event start with '#'
                        if event['href'] == '#':
                            event_id = quote_plus(id_)
                        else:
                            event_id = event['href'].split("'")[1]
                            event_id = quote_plus(event_id)
                        event_url = f"http://sjw.history.go.kr/search/inspectionDayList.do?treeID={event_id}"
                        
                        # Try / Except => Unknown error occur sometime
                        try:
                            event_page = BeautifulSoup(requests.get(event_url).text, 'lxml')
                        except:
                            print(f"{king_area_}_{month.text}_{event.text} Error | Restarting...")
                            time.sleep(100)
                            event_page = BeautifulSoup(requests.get(event_url).text, 'lxml')
                        time.sleep(0.2)

                        book_month_page = event_page.find("div", {'class': 'view_tit'})
                        book_section_text = book_month_page.find('span').text.strip()
                        event_time = book_section_text.split("\r")[0] + book_section_text.split("\t")[4]
                        date = " ".join(event_time.split()[3:7])

                        # Blank page processing
                        try:
                            name_text = event_page.find('h2', {'class': 'search_tit'}).text
                        except AttributeError: # Because of NoneType
                            continue
                        
                        # Empty content processing
                        if name_text == "":
                            name_text = "\t"

                        event_chinese = event_page.find('ul', {'class': 'sjw_list'})

                        for paragraph_chinese in event_chinese.findAll('a'):
                            # Blank page processing
                            try:
                                chinese_text = paragraph_chinese.find('span').text
                            except AttributeError: # Because of NoneType
                                continue
                            for chinese_char in chinese_text.split('○'):
                                hanja = chinese_char
                                if not hanja == "":
                                    hanja = " ".join(chinese_char.split())
                                    articles.append({'hanja': hanja, 'date': date})

                    pickle.dump(articles, fwrite)

                time.sleep(3)