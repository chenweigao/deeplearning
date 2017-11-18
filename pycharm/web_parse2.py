from bs4 import BeautifulSoup

file = 'C:/Users/weigao/Videos/Plan-for-combating-master/week1/1_2/1_2answer_of_homework/1_2_homework_required/index.html'

with open(file, 'r') as web_data:
    Soup = BeautifulSoup(web_data, 'lxml')
    titles = Soup.select(
        'body > div > div > div.col-md-9 > div > div > div > div.caption > h4 > a')
    images = Soup.select(
        'body > div > div > div > div > div > div > img')
    rates = Soup.select('body > div > div > div > div > div > div > div > h4')

    stars = Soup.select(
        'body > div > div > div.col-md-9 > div > div > div > div.ratings > p:nth-of-type(2)')
    # print(titles, sep='\n-----------------------------------\n ')
info = []
for title, image, rate, star in zip(titles, images, rates, stars):
    data = {
        'title': title.get_text(),
        'image': image.get_text('src'),
        'rate': rate.get_text(),
        'star': len(star.find_all("span", 'glyphicon glyphicon-star'))
    }
    info.append(data)

for _ in info:
    if(_['star'] >= 4):
        print(_['title'])
