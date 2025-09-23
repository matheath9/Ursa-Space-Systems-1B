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



# Analyze the class distrubution to ensure labels are balanced

unique, counts = np.unique(Y_train, return_counts = True)
class_distrubution = (dict(zip(unique, counts)))
print("The class distribution is: ", class_distrubution)



# Asses basic global statistics for  dataset
X_float = X_train.astype(np.float32)   # convert to float just in case

overall_mean = X_float.mean()
overall_std = X_float.std()

print("Overall mean value is: ", overall_mean)
print("Overall standard deviation is: %.4f" % overall_std)

# Normalization for better convergence when training
X_norm = X_train / overall_std




# Check for brightness bias
classes = np.unique(Y_train)

for cls in classes:
    class_pixels = X_train[Y_train == cls]
    mean_brightness = class_pixels.mean(axis=(1,2))  # per-image mean
    print(f"Class {cls}: avg brightness = {mean_brightness.mean():.4f} ± {mean_brightness.std():.4f}")



# Check for contrast or variance bias
for cls in classes:
    class_pixels = X_train[Y_train == cls]
    contrast = class_pixels.std(axis=(1,2))
    print(f"Class {cls} avg contrast= {contrast.mean():.4f} ± {contrast.std():.4f}")



