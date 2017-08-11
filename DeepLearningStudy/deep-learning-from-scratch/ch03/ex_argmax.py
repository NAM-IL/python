# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
import pickle
from dataset.mnist import load_mnist
from common.functions import sigmoid, softmax


a = np.array([[[1, 5, 3, 8], [5, 2, 7, 4]], [[5, 2, 7, 4], [1, 5, 3, 8]] ])

print("a:{}".format(a)) 

print("argmax(a,axis=0):{}".format(np.argmax(a, axis=0)))
print("argmax(a,axis=1):{}".format(np.argmax(a, axis=1)))
print("argmax(a,axis=2):{}".format(np.argmax(a, axis=2)))

sample = np.random.random((2,3,4))*10
print("sample:\n{}".format(sample)) 
print("sample.shape:\n{}".format(sample.shape)) 

# argmax_indices = np.argmax(sample, axis=2)
print("argmax(sample,axis=0):\n{}".format(np.argmax(sample, axis=0)))
print("argmax(sample,axis=1):\n{}".format(np.argmax(sample, axis=1)))
print("argmax(sample,axis=2):\n{}".format(np.argmax(sample, axis=2)))
