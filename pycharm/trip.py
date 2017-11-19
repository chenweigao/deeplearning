from bs4 import BeautifulSoup 
import requests 

url = 'https://cn.tripadvisor.com/Attractions-g60763-Activities-New_York_City_New_York.html'
wb_data = requests.get(url)
soup = BeautifulSoup(wb_data.text,'lxml')
titles = soup.select('div.item.name > a')
cates = soup.select('div.prw_rup prw_common_location_rating_simple')

for title,cate in zip(titles,cates): 
    data = {
        'title':title.get_text(),
        'cate':list(cate.stripped_strings) 
    }
    print(data)