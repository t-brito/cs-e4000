import gzip
import os
import numpy as np
import urllib.request

DATA_URL = 'http://yann.lecun.com/exdb/mnist/'
TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'
TRAIN_LABELS = 'train-labels-idx1-ubyte.gz'
TRAIN_SIZE = 60000
TEST_IMAGES = 't10k-images-idx3-ubyte.gz'
TEST_LABELS = 't10k-labels-idx1-ubyte.gz'
TEST_SIZE = 10000

NUM_CHANNELS = 1
IMAGE_COLS = 28
IMAGE_ROWS = 28

def _maybe_download(filename):
    filepath = os.path.join('data', filename)
    if not os.path.exists(filepath):
        urllib.request.urlretrieve(DATA_URL+filename, filepath)
    return filepath

def _extract_labels(filename, num_images):
    with gzip.open(_maybe_download(filename)) as flbl:
        flbl.read(8)
        label = np.fromstring(flbl.read(), dtype=np.int8)
    return label

def _extract_images(filename, num_images):
    with gzip.open(_maybe_download(filename), 'rb') as fimg:
        fimg.read(16)
        image = np.fromstring(fimg.read(), dtype=np.uint8).reshape(num_images, IMAGE_ROWS, IMAGE_COLS)
    return image

def load_data():
    '''returns (train_img, train_lbl), (test_img, test_lbl)'''
    train_img = _extract_images(TRAIN_IMAGES,TRAIN_SIZE)
    train_img = train_img.reshape(train_img.shape[0], NUM_CHANNELS, IMAGE_ROWS, IMAGE_COLS).astype('float32') / 255
    train_lbl = _extract_labels(TRAIN_LABELS,TRAIN_SIZE)
    test_img = _extract_images(TEST_IMAGES,TEST_SIZE)
    test_img = test_img.reshape(test_img.shape[0], NUM_CHANNELS, IMAGE_ROWS, IMAGE_COLS).astype('float32') / 255
    test_lbl = _extract_labels(TEST_LABELS,TEST_SIZE)

    return (train_img, train_lbl), (test_img, test_lbl)

if __name__ == '__main__':
    (train_img, train_lbl), (test_img, test_lbl) = load_data()
    print("train_img shape:", train_img.shape)
    print("train_lbl shape:", train_lbl.shape)
    print("test_img shape:", test_img.shape)
    print("test_lbl shape:", test_lbl.shape)
