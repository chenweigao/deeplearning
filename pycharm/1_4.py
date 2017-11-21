import requests
from bs4 import BeautifulSoup

url = 'http://zhuanzhuan.58.com/detail/932582237472604166z.shtml'

web_data = requests.get(url)
soup = BeautifulSoup(web_data.text,'lxml')

title = soup.title.text
price = soup.select('span.price_now > i')[0].text
area = soup.select('div.palce_li > span')[0].text
print(title,price,area)

