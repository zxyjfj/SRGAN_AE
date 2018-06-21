import os
from glob import glob

from skimage import io


def ProcessingImage(filename):
    image = io.imread(filename)

    new_image = image[:216, :176, :]

    save_filename = '%s.png' % (os.path.basename(filename).split('.')[0])

    io.imsave('../data/train/' + save_filename, new_image)


if __name__ == '__main__':
    file_list = glob('../data/imgs/*.jpg')
    for filename in file_list:
        print(filename)
        ProcessingImage(filename)
