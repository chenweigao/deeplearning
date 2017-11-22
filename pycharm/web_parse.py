from bs4 import BeautifulSoup
info = []
with open('C:/Users/Administrator/Videos/Plan-for-combating-master/week1/1_2/1_2code_of_video/web/new_index.html', 'r') as web_data:
    Soup = BeautifulSoup(web_data, 'lxml')
    images = Soup.select('body > div.main-content > ul > li > img')
    titles = Soup.select('ul > li > div.article-info > h3 > a')
    pics = Soup.select('ul > li > img')
    descs = Soup.select('ul > li > div.article-info > p.description')
    rates = Soup.select('ul > li > div.rate > span')
    cates = Soup.select('ul > li > div.article-info > p.meta-info')
    # print(images,titles,pics,descs,rates,cates,sep='\n-----------------\n')


for title, image, desc, rate, cate in zip(titles, images, descs, rates, cates):
    data = {
        'title': title.get_text(),
        'rate': rate.get_text(),
        'cate': list(cate.stripped_strings),
        'image': image.get('src')
    }
    info.append(data)

for i in info:
    if float(i['rate']) > 3:
        print(i['title'], i['cate'])
