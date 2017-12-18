import scipy.misc
import matplotlib.pyplot as plt
import numpy as np

face = scipy.misc.face()[:,range(768)]
gray=face[:,:,1]

def get_indices(size):
    arr = np.arange(size)
    return arr % 4 == 0


face_copy1 = face.copy()
row_indices = get_indices(face.shape[0])
col_indices = get_indices(face.shape[1])
face_copy1[row_indices, col_indices] = 0

plt.subplot(211)
plt.imshow(face_copy1)

face_copy2 = face.copy()
face_copy2[(gray > gray.max() / 4) & (gray < 3 * gray.max() / 4)] = 0
plt.subplot(212)
plt.imshow(face_copy2)

plt.show()
