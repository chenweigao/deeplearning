import scipy.misc
import matplotlib.pyplot as plt
import numpy as np
face = scipy.misc.face()
row_max = face.shape[0]
col_max = face.shape[1]

def shuffle_indices(size):
    arr = np.arange(size)
    np.random.shuffle(arr)

    return arr

row_indices = shuffle_indices(row_max)
# print(row_indices,len(row_indices))
np.testing.assert_equal(len(row_indices), row_max)

col_indices = shuffle_indices(col_max)
np.testing.assert_equal(len(col_indices), col_max)

plt.imshow(face[np.ix_(row_indices, col_indices)])
plt.show()