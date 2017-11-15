from bs4 import BeautifulSoup

file = 'C:/Users/Administrator/Videos/Plan-for-combating-master/week1/1_2/1_2answer_of_homework/1_2_homework_required/index.html'

with open(file, 'r') as web_data: 
    Soup = BeautifulSoup(web_data, 'lxml')
    titles = Soup.select(
        'body > div > div > div.col-md-9 > div > div > div > div.caption > h4 > a')
    images = Soup.select(
        'body > div > div > div > div > div > div > img')
    rates = Soup.select('body > div > div > div > div > div > div > div > h4')

    print(titles,sep='\n-----------------------------------\n ')





    # ''
    # 'body > div:nth-child(2) > div > div.col-md-9 > div:nth-child(2) > div:nth-child(8) > div > div.caption > h4:nth-child(2) > a'