from bs4 import BeautifulSoup
import requests
url = 'http://bj.xiaozhu.com/fangzi/1508951935.html'
web_data = requests.get(url)
soup = BeautifulSoup(web_data.text, 'lxml')

title = soup.select('div.pho_info > h4')[0].text
address = soup.select('div.pho_info > p')[0].get('title')
price = soup.select('div.day_l > span')[0].text
prc = soup.select('#curBigImage')[0].get('src')
print(price, prc)

host_name = soup.select('a.lorder_name')[0].text
host_gender = soup.select('div.member_pic > div')[0].get('class')[0]
print(host_gender)


def print_gender(class_name):
    if class_name == member_ico1:
        return 'woman'
    if class_name == member_ico:
        return 'man'


data = {
    'title': title,
    'address': address,
    'price': price,
    'host_name': host_name,
    'host_gender': host_gender
}

print(data)

page_link = []


def get_page_link(page_number):
    for each_number in range(1, page_number):
        full_url = 'http://bj.xiaozhu.com/search-duanzufang-p{}-0'.format(
            each_number)
        web_data = requests.get(full_url)
        soup = BeautifulSoup(web_data, 'lxml')
        for link in soup.select(a.resule_img_a):
            page_link.append(link)
