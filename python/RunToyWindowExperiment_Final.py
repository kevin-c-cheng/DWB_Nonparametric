# -*- coding: utf-8 -*-
"""
Created on Fri May  6 21:20:40 2022

@author: Kevin Cheng
"""
import numpy as np
import scipy.io as scio
from scipy.special import erfinv
import ot

def GaussWassBary(X, Q):
    if (len(np.shape(Q))==1):
        Q = np.expand_dims(Q,0)
    Qm = Q[:,0]
    QS = Q[:,1]
    tmp = (X@np.sqrt(QS))**2
    return np.transpose(np.concatenate((np.expand_dims(X@Qm,0), np.expand_dims(tmp,0))))


def EvaluateEmpGauss(obs, mean, cov, nSamp=10000):
    n = len(obs)
    samps = np.random.normal(mean, cov, nSamp )
    a = 1/nSamp*np.ones(nSamp)
    b = 1/n*np.ones(n)
    M = ot.dist(np.expand_dims(samps,1), np.expand_dims(obs,1), metric='sqeuclidean')
    C = ot.emd(a,b,M)
    return np.sum(M*C)

def EvaluateEmpGauss2(obs, mean, cov, nSampMult=1000):
    n = len(obs)
    samps = np.sort(np.random.normal(mean, np.sqrt(cov), nSampMult*n ))
#    obs = np.sort(obs)
#    samps2 = np.mean(np.reshape(samps, [n, nSampMult]),1)
    obs = np.repeat(np.sort(obs),nSampMult)
    return 1/(n*nSampMult)*np.sum(np.square(obs-samps))

def EvaluateEmpGaussQuant2(obs, mean, cov, nQuantMin=1e4):
    n = len(obs)
    nQuant = int(np.ceil(nQuantMin/n))*n
    
    quant = np.linspace(0.5/nQuant, 1-0.5/nQuant, nQuant)
    quantVal = mean + np.sqrt(cov)*np.sqrt(2)*erfinv(2*quant-1)
    obs2 = np.sort(obs)# May have to throw out one or two samples here for this to come out even
    return 1/(nQuant)*np.sum(np.square(np.repeat(obs2, int(nQuant/n))-quantVal))

#rAll = [50, 100, 200, 400, 800]
#nAll = np.linspace(10,50,9).astype(int)
#nAll = np.linspace(0.02,0.5,25)
rAll = [4, 10, 20, 50, 80, 100, 150, 200, 250, 300, 350, 400, 500, 600, 700, 800] # n must be even
nAll = [4, 6, 8, 10, 12, 14, 16, 18, 20, 24, 28, 30, 34, 38, 40, 44, 48, 50, 54, 58, 60, 64, 70, 74, 80, 84, 90, 94, 100, 110, 120, 130, 140, 150, 200, 210, 220, 230, 240, 250, 300, 350, 400] # n must be even
#nAll = np.linspace(4,400,23).astype(int)
tAll = [0.5]

m1 = 0
m2 = 10
s1 = 5
s2 = 0.2

nIter = int(5000)
nSamp = int(1e5)
nSampMult = int(1e4)
out = np.zeros((len(rAll), len(nAll), len(tAll), nIter))

# System 1
for rInd in range(len(rAll)):
    r=rAll[rInd]
    print('r: ', str(r))
    for tInd in range(len(tAll)):
        t = tAll[tInd]
        mHalf = (1-t)*m1 + t*m2
        sHalf = ((1-t)*np.sqrt(s1)+t*np.sqrt(s2))**2
        print('    t: ', str(t))
        for nInd in range(len(nAll)):
            n = nAll[nInd]
            print('        n: ', str(n))

            for i in range(nIter):
                win = np.zeros(n)
                for j in range(-int(n/2), int(n/2)): # we want this to be -1.5, -0.5, 0.5, 1.5, etc so we offset by half in next line
                    w = t + (j+0.5)/r
                    w = np.clip(w,0,1)
                    mb = (1-w)*m1 + w*m2
                    sb = ((1-w)*np.sqrt(s1) + w*np.sqrt(s2))**2
                    win[int(j+n/2)] = np.random.normal(mb, np.sqrt(sb))
#                out[rInd,nInd,tInd,i] = EvaluateEmpGauss2(win, mHalf, sHalf, nSampMult)
                out[rInd,nInd,tInd,i] = EvaluateEmpGaussQuant2(win, mHalf, sHalf)
#            scio.savemat('WindowExperimentF.mat', mdict={'out':out, 'r':r, 'n':n, 'win':win})

## Constant xt
mHalf = 0.5*m1 + 0.5*m2
sHalf = (0.5*np.sqrt(s1)+0.5*np.sqrt(s2))**2

nAll2 = np.linspace(4,400,198+1).astype(int) # n must be even

outC = np.zeros((len(nAll2), nIter))
for nInd in range(len(nAll2)):
    n = nAll2[nInd]
    print('        n: ', str(n))
    for i in range(nIter):
        #win = np.zeros(n)
        win = np.random.normal(mHalf, np.sqrt(sHalf), size=n)
        outC[nInd, i] = EvaluateEmpGaussQuant2(win, mHalf, sHalf)

## Dynamic xt
outD = np.zeros((len(nAll2), nIter))
r=300
for nInd in range(len(nAll2)):
    n = nAll2[nInd]
    print('        n: ', str(n))
    for i in range(nIter):
        win = np.zeros(n)
        for j in range(-int(n/2), int(n/2)): # we want this to be -1.5, -0.5, 0.5, 1.5, etc so we offset by half in next line
            w = t + (j+0.5)/r
            w = np.clip(w,0,1)
            mb = (1-w)*m1 + w*m2
            sb = ((1-w)*np.sqrt(s1) + w*np.sqrt(s2))**2
            win[int(j+n/2)] = np.random.normal(mb, np.sqrt(sb))
        outD[nInd,i] = EvaluateEmpGaussQuant2(win, mHalf, sHalf)



# System 2
f1 = 0.1
f2 = 0.9
mb1 = (1-f1)*m1 + f1*m2
sb1 = ((1-f1)*np.sqrt(s1) + f1*np.sqrt(s2))**2
mb2 = (1-f2)*m1 + f2*m2
sb2 = ((1-f2)*np.sqrt(s1) + f2*np.sqrt(s2))**2

m1=mb1
m2=mb2
s1=sb1
s2=sb2

out2 = np.zeros((len(rAll), len(nAll), len(tAll), nIter))

for rInd in range(len(rAll)):
    r=rAll[rInd]
    print('r: ', str(r))
    for nInd in range(len(nAll)):
        n = nAll[nInd]
        print('    n: ', str(n))
        for tInd in range(len(tAll)):
            t = tAll[tInd]
            mHalf = (1-t)*m1 + t*m2
            sHalf = ((1-t)*np.sqrt(s1)+t*np.sqrt(s2))**2
            print('    t: ', str(t))

            for i in range(nIter):
                win = np.zeros(n)
                for j in range(-int(n/2), int(n/2)): # we want this to be -1.5, -0.5, 0.5, 1.5, etc so we offset by half in next line
                    w = t + (j+0.5)/r
                    w = np.clip(w,0,1)
                    mb = (1-w)*m1 + w*m2
                    sb = ((1-w)*np.sqrt(s1) + w*np.sqrt(s2))**2
                    win[int(j+n/2)] = np.random.normal(mb, np.sqrt(sb))
#                out2[rInd,nInd,tInd,i] = EvaluateEmpGauss2(win, mHalf, sHalf, nSampMult)
                out2[rInd,nInd,tInd,i] = EvaluateEmpGaussQuant2(win, mHalf, sHalf)

scio.savemat('WindowExperimentF.mat', mdict={'out2':out2, 'out':out, 'outD':outD, 'outC':outC, 'window':nAll, 'window2':nAll2, 'rAll':rAll, 'nAll':nAll})
                        
    