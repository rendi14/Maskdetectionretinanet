import sys
sys.path.insert(0, '../')

# import keras_retinanet
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color
from keras_retinanet.utils.gpu import setup_gpu


# import miscellaneous modules
from pygame import mixer

import numpy as np
import os
import cv2

from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color

from keras.models import load_model
from keras_retinanet import models

# set tf backend to allow memory to grow, instead of claiming everything
import tensorflow as tf

# use this to change which GPU to use
gpu = "0,"

# set the modified tf session as backend in keras
setup_gpu(gpu)

# ## Load RetinaNet model

# In[ ]:
mixer.init()
sound = mixer.Sound('alarm.wav')
cap = cv2.VideoCapture(0)
model_path = os.path.join('..', 'snapshots', 'resnet50_csv_50.h5')

# load retinanet model
model = models.load_model(model_path, backbone_name='resnet50')
model = models.convert_model(model)

# print(model.summary())
labels_to_names = {0: 'mask', 1: 'no_mask'}

def predict(image):
    image = preprocess_image(image.copy())
    image, scale = resize_image(image)

    boxes, scores, labels = model.predict_on_batch(
     np.expand_dims(image, axis=0)
     )

    boxes /= scale

    return boxes, scores, labels

umbralScore = 0.4

# get the reference to the webcam
camera = cv2.VideoCapture(0)
camera_height = 500

while (True):
    # read a new frame
    _, frame = camera.read()

    boxes, scores, labels = predict(frame)

    draw = frame.copy()

    for box, score, label in zip(boxes[0], scores[0], labels[0]):
        # if score > 0.5:
        #     # print(box)
        #     b = box.astype(int)
        #     if label == 0:
        #         color = [0, 255, 0]
        #
        #     elif label == 1:
        #         color = [0, 0, 255]
        if (score > 0.5):
            # person is feeling sleepy so we beep the alarm
            b = box.astype(int)
            if label == 0:
                color = [0, 255, 0]

            elif label == 1:
                color = [0, 0, 255]
                sound.play()
            draw_box(draw, b, color=color)
            caption = "{} {:.1f}".format(labels_to_names[label], score)
            draw_caption(draw, b, caption.upper())
    # show the frame
    cv2.imshow("Test out", draw)

    key = cv2.waitKey(100)

    # quit camera if 'q' key is pressed
    if key & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()