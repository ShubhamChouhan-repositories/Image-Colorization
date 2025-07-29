#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import argparse
import cv2
import os


# In[2]:


DIR = r"C:\Users\shubh\Documents\colorize"
PROTOTXT = os.path.join(DIR, "model/colorization_deploy_v2.prototxt")
POINTS = os.path.join(DIR, "model/pts_in_hull.npy")
MODEL = os.path.join(DIR, "model/colorization_release_v2.caffemodel")
IMAGE_PATH = os.path.join(DIR, "images", "nature.jpg")


# In[3]:


args = {"image": IMAGE_PATH}

# Validate paths
if not os.path.exists(PROTOTXT):
    raise FileNotFoundError(f"Prototxt file not found: {PROTOTXT}")
if not os.path.exists(MODEL):
    raise FileNotFoundError(f"Model file not found: {MODEL}")
if not os.path.exists(POINTS):
    raise FileNotFoundError(f"Points file not found: {POINTS}")
if not os.path.exists(args["image"]):
    raise FileNotFoundError(f"Input image not found: {args['image']}")


# In[4]:


print("Load model")
net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)
pts = np.load(POINTS)


# In[5]:


class8 = net.getLayerId("class8_ab")
conv8 = net.getLayerId("conv8_313_rh")
pts = pts.transpose().reshape(2, 313, 1, 1)
net.getLayer(class8).blobs = [pts.astype("float32")]
net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]


# In[6]:


image = cv2.imread(args["image"])
if image is None:
    raise ValueError(f"Failed to load image: {args['image']}")

scaled = image.astype("float32") / 255.0
lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)

resized = cv2.resize(lab, (224, 224))
L = cv2.split(resized)[0]
L -= 50


# In[7]:


print("Colorizing the image")
net.setInput(cv2.dnn.blobFromImage(L))
ab = net.forward()[0, :, :, :].transpose((1, 2, 0))

ab = cv2.resize(ab, (image.shape[1], image.shape[0]))

L = cv2.split(lab)[0]
colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)

colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
colorized = np.clip(colorized, 0, 1)

colorized = (255 * colorized).astype("uint8")


# In[8]:


cv2.imshow("Original", image)
cv2.imshow("Colorized", colorized)
cv2.waitKey(0)
cv2.destroyAllWindows()





