# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
import pickle
from dataset.mnist import load_mnist
from common.functions import sigmoid, softmax


# myInfo = {"weight" : 60, "height" : 170}
# print(myInfo)
# print("height:{}".format(myInfo["height"]))

def dumpInfo(paramInfo):
    with open('my_info.pkl', 'wb') as handle_dump:
        pickle.dump(paramInfo, handle_dump, protocol=pickle.HIGHEST_PROTOCOL)
   
def loadInfo(): 
    with open('my_info.pkl', 'rb') as handle_load:
        myInfo = pickle.load(handle_load)
        print("myInfo: {}".format(myInfo))
    
    return myInfo


myInfo = loadInfo()

myInfo["gender"] = "male"

dumpInfo(myInfo)
myInfo = loadInfo()

