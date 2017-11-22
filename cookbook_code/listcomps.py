symbols = '$@#$$!@#'
codes = [ord(symbol) for symbol in symbols]
print(codes)
# 这是列表推导(list comprehension)的正确表达方法

beyond_ascii_1 = [ord(_) for _ in symbols if ord(_) >= 36]
beyond_ascii_2 = list(filter(lambda c: c >= 36, map(ord, symbols)))
print(beyond_ascii_1, beyond_ascii_2)
# 比较列表推导和map/filter组合

colors = ['black', 'white']
sizes = ['S', 'M', 'L']
Tshits = [(color, size)  for size in sizes for color in colors]
print(Tshits)
