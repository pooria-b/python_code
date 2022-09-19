import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.datasets import mnist
from tqdm.notebook import tqdm
import cv2

(X_train, y_train), (X_test, y_test) = mnist.load_data()
print("The size of training dataset is: ", X_train.shape)
print("The size of testing dataset is: ", X_test.shape)


def number_to_image(input_number):
  input_num_picture = []
  for i in tqdm(range(len(y_test))):
 
  # picture extraction
    if (input_number == y_test[i]):
      input_num_picture.append(i)
  img_num = random.choice(input_num_picture)
  
  # pre processing on image
  resized_img = cv2.resize(X_test[img_num], (450,450))
  
  return resized_img