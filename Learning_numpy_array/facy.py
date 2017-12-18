import scipy.misc
import matplotlib.pyplot as plt

face = scipy.misc.face()

row_max = face.shape[0]
col_max = face.shape[1]
row_mean = [(int)(i*(row_max/col_max)) for i in range(col_max)]
# face[row_mean, range(col_max)] = 0

# print(len(row_mean))
# 1024
col_mean = [(int)(i*col_max/row_max) for i in range(row_max)]
# print(len(col_mean))
# 768
face[range(row_max), col_mean] = 0

face[row_mean, range(col_max - 1, -1, -1)] = 0

plt.imshow(face)
plt.show()