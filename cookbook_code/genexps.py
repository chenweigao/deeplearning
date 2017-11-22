symbols = '$@#$$!@#'
beyond_ascii = tuple(ord(_) for _ in symbols)
print(beyond_ascii)
#生成器表达式(generator expression)

colors = ['black', 'white']
sizes = ['S', 'M', 'L']
for _ in ('%s %s' % (size,color) for size in sizes for color in colors):
    print(_)
#生成器的方法相比较与上一个，它的表达式逐个产生元素，避免了额外的内存占用