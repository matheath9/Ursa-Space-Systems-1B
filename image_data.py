import numpy as np

npz = np.load(r'C:\Users\matan\.cache\kagglehub\datasets\saurabhbagchi\ship-and-iceberg-images\versions\1\input_data.npz')

X_train = npz['X_train']
Y_train = npz['Y_train']
del npz

print('We have {} examples to work with'.format(Y_train.shape[0]))


import matplotlib.pyplot as plt

ix=2
plt.imshow(np.squeeze(X_train[ix, :, :, 2]), cmap='gray')
plt.title(f"This is: {labels[int(Y_train[ix])]}")
plt.axis('off')
plt.show()