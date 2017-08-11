# coding: utf-8
import numpy as np
import matplotlib.pylab as plt
from numba.typeconv.rules import _init_casting_rules
import random

class errorFunTest:
    
    def __init__(self):
        self.y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
        self.t = []
    
    def initVariable(self):
        _t = np.random.rand(1)*10
        tIdx =  _t.astype(np.int8)
        self.t = np.zeros(10).astype(np.int8)
        self.t[tIdx] = 1 # 정답 레이블 : one-hot encoding
        self.y = random.sample( self.y, len(self.y)) # 추정 값
        self.totalSum = np.sum(self.y)
        
    def setCorrect(self):
        idx = np.argmax(self.y)
        self.t = np.zeros(10).astype(np.int8)
        self.t[idx] = 1
    
    # 평균 제곱 오차(MSE: Mean squared error)
    def mean_square_error(self, y, t):
        return 0.5*np.sum((y-t)**2)
    
    # 표차 엔트로피 오차(CEE: Cross entropy error)
    def cross_entropy_error(self, y, t):
        delta = 1e-7
        return -np.sum(t * np.log(y + delta))
    
    def printVariables(self):
        print('totalSum = {0}'.format(self.totalSum))
        print('t = {0}'.format(self.t))
        print('y = {0}'.format(self.y))


ex1 = errorFunTest()
ex2 = errorFunTest()

# ex1.initVariable()
# MSE1 = ex1.mean_square_error(np.array(ex1.y), np.array(ex1.t))
# CEE1 = ex1.cross_entropy_error(np.array(ex1.y), np.array(ex1.t))
# ex1.printVariables()
# # print('MSE_1 = {0}'.format(MSE1))
# print('CEE_1 = {0}'.format(CEE1))
# 
# ex2.initVariable()
# ex2.setCorrect()
# MSE2 = ex2.mean_square_error(np.array(ex2.y), np.array(ex2.t))
# CEE2 = ex2.cross_entropy_error(np.array(ex2.y), np.array(ex2.t))
# ex2.printVariables()
# # print('MSE_2 = {0}'.format(MSE2))
# print('CEE_2 = {0}'.format(CEE2))


batch = np.random.choice(60000, 10)
print(">> np.random.choice(60000, 10)")
print('batch = {0}'.format(batch))

