from collections import namedtuple

City = namedtuple('City','name country population coordinates')
xian = City('xian','CN','5300',(34.123,389.231))
print(xian)

print(City._fields)