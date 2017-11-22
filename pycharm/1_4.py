import requests
from bs4 import BeautifulSoup

url = 'http://zhuanzhuan.58.com/detail/932582237472604166z.shtml'


def get_item_info(url):

    web_data = requests.get(url)
    soup = BeautifulSoup(web_data.text,'lxml')

    data = {
        'title':soup.title.text,
        'price':soup.select('span.price_now > i')[0].text,
        'area':soup.select('div.palce_li > span')[0].text
    }
    print(data)

# def get_links_from():
#     urls = []
#     list_view ='http://xa.58.com/yamaha/?PGTID=0d100000-001e-35c3-a6f6-339c2ed564f7&ClickID=0'
#     wb_date =  requests.get(list_view)
#     soup = BeautifulSoup(wb_date,'lxml')
#     for link in soup.select('t.a'):
#         urls.append(link.get('herf').split('?')[0])
#     # return urls
#         print(urls)
# get_links_from()