import struct
import numpy as np
import os

def read_idx(filename):
    with open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)

# Define the path to the MNIST_ORG folder
mnist_folder = 'MNIST_ORG'

# Load training data
train_images_path = os.path.join(mnist_folder, 'train-images.idx3-ubyte')
train_labels_path = os.path.join(mnist_folder, 'train-labels.idx1-ubyte')

train_images = read_idx(train_images_path)
train_labels = read_idx(train_labels_path)

# Load test data
test_images_path = os.path.join(mnist_folder, 't10k-images.idx3-ubyte')
test_labels_path = os.path.join(mnist_folder, 't10k-labels.idx1-ubyte')

test_images = read_idx(test_images_path)
test_labels = read_idx(test_labels_path)

# Print the shapes to verify
print(f'Training images shape: {train_images.shape}')  # Should print (60000, 28, 28)
print(f'Training labels shape: {train_labels.shape}')  # Should print (60000,)
print(f'Test images shape: {test_images.shape}')        # Should print (10000, 28, 28)
print(f'Test labels shape: {test_labels.shape}')        # Should print (10000,)
