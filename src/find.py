import os.path
from glob import glob

import cv2


def detect(filename, cascade_file="haarcascade_frontalface_alt.xml"):
    if not os.path.isfile(cascade_file):
        raise RuntimeError("%s: not found" % cascade_file)

    cascade = cv2.CascadeClassifier(cascade_file)
    image = cv2.imread(filename)
    if image == None:  # read failure and skip this image
        return
    if image.shape[2] == 1:  # drop gray images
        return
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    faces = cascade.detectMultiScale(
        gray,
        # detector options
        scaleFactor=1.1,
        minNeighbors=3,
        minSize=(96, 96))
    for i, (x, y, w, h) in enumerate(faces):
        face = image[y:y + h, x:x + w, :]
        face = cv2.resize(face, (128, 128))
        save_filename = '%s.png' % (os.path.basename(filename).split('.')[0])
        cv2.imwrite("../data/train/" + save_filename, face)


if __name__ == '__main__':
    file_list = glob('../data/imgs/*.jpg')
    for filename in file_list:
        print(filename)
        detect(filename)
