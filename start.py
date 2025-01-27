from mnist import MNIST
import numpy as np 
from lib import *

mndata = MNIST('./data')
images, labels = mndata.load_training()
t_images, t_labels = mndata.load_testing()

images = np.array(images)
t_images = np.array(t_images)
labels = np.array(labels)
t_labels = np.array(t_labels)

print_statistics(images, t_images, labels, t_labels)

# add bias to the data (train & test)
(m,n) = images.shape
bias = np.ones((m,1))
images = np.hstack((bias,images))
(m2,n2) = t_images.shape
t_bias = np.ones((m2,1))
t_images = np.hstack((t_bias,t_images))
