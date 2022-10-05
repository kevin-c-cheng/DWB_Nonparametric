# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 11:28:47 2022

@author: Kevin Cheng
"""
import numpy as np
import scipy.io as scio
import scipy.stats as scistats
import os


def ProcessPDF(supp,pdf,minVal=-100):
    eps = 1e-10
    n = len(pdf)
    cdf = np.cumsum(pdf)
    quant = np.zeros(n)
    for i in range(n):
        idx = np.argwhere(cdf >=(i+1)/n-eps)[0] 
        quant[i] = supp[idx]
    
    return (cdf, quant)
    
np.random.seed(0)
nSupp = 100000
supp = np.linspace(-5,5,nSupp)
suppQ = np.linspace(0,1,nSupp)
dx = (max(supp)-min(supp))/len(supp)

pdf1 = scistats.norm.pdf(supp, -3, 0.5) + scistats.norm.pdf(supp, 3, 0.5)
pdf1 = pdf1/sum(pdf1)
(cdf1, quant1) = ProcessPDF(supp, pdf1)

pdf2 = abs(supp)<4;
pdf2 = pdf2/sum(pdf2)
(cdf2, quant2) = ProcessPDF(supp, pdf2)

nSamp = 5
samp3  = np.random.choice(np.linspace(0,len(supp), len(supp)).astype(int), nSamp)
pdf3 = np.zeros(len(supp))
for i in range(5):
    pdf3 = pdf3 + scistats.norm.pdf(supp, supp[samp3[i]], 1e-4)   
#pdf3[samp3] = 1/nSamp
pdf3 = pdf3/sum(pdf3)
(cdf3, quant3) = ProcessPDF(supp, pdf3)
foldOut = 'C:/Users/Kevin Cheng/Box Sync/SimplexRandomWalk/TimeSeriesClustering/oneDimensional/data/TimeSeries/'
scio.savemat(foldOut+'PureState.mat', mdict={'pdf1':pdf1, 'pdf2':pdf2, 'pdf3':pdf3,
                                    'cdf1':cdf1, 'cdf2':cdf2, 'cdf3':cdf3,  
                                    'quant1':quant1, 'quant2':quant2, 'quant3':quant3,
                                    'supp':supp, 'suppQ':suppQ})  

## Interpolated model
freqAll = [1, 2]
sampRateAll = [100, 200, 300, 150, 250]

fold = 'C:/Users/Kevin Cheng/Box Sync/SimplexRandomWalk/TimeSeriesClustering/oneDimensional/data/SimplexTrajectories/'
foldOut = 'C:/Users/Kevin Cheng/Box Sync/SimplexRandomWalk/TimeSeriesClustering/oneDimensional/data/TimeSeries/'
try:
    os.mkdir(foldOut)
except:
    print('fold exists')

for freq in freqAll:
    for rate in sampRateAll:
        for f in os.listdir(fold):
            Y = np.array([])
            d = scio.loadmat(fold+f)
            try:
                X = d['X']
            except:
                X = d['x']

            continuousT = np.linspace(0,9,freq*len(X))
            discreteX = np.linspace(0,9, (rate*9))
            X1 = np.interp(discreteX, continuousT, np.tile(X[:,0], freq))
            X2 = np.interp(discreteX, continuousT, np.tile(X[:,1], freq))
            X3 = np.interp(discreteX, continuousT, np.tile(X[:,2], freq))
            T = len(X1)
            for i in range(T):
                quantOut = X1[i]*quant1 + X2[i]*quant2 + X3[i]*quant3
                window = np.random.choice(quantOut, 1, True)
                Y = np.append(Y, window)
        
            X = np.zeros((len(X1), 3))
            X[:,0]=X1
            X[:,1]=X2
            X[:,2]=X3
            
            d['X'] = X
            d['Y'] = Y
            d['K'] = 3
            d['T'] = len(Y)
            d['pdf1'] = pdf1
            d['pdf2'] = pdf2
            d['pdf3'] = pdf3
            d['quant1'] = quant1
            d['quant2'] = quant2
            d['quant3'] = quant3
            scio.savemat(foldOut+f[:-4]+'_blurred_rate'+str(rate)+'_freq'+str(freq)+'.mat', mdict=d)  
                

### Run Multiple iterations
#sampRateAll = [150, 200, 350]
sampRateAll = [150, 200]
nIter = 500

f='vertex.mat'
d = scio.loadmat(fold+f)
try:
    X = d['X']
except:
    X = d['x']

freq=1

for it in range(nIter):
    for rate in sampRateAll:
        continuousT = np.linspace(0,9,freq*len(X))
        discreteX = np.linspace(0,9, (rate*9))
        X1 = np.interp(discreteX, continuousT, np.tile(X[:,0], freq))
        X2 = np.interp(discreteX, continuousT, np.tile(X[:,1], freq))
        X3 = np.interp(discreteX, continuousT, np.tile(X[:,2], freq))
        T = len(X1)
        
        Y = np.array([])
        for i in range(T):
            quantOut = X1[i]*quant1 + X2[i]*quant2 + X3[i]*quant3
            window = np.random.choice(quantOut, 1, True)
            Y = np.append(Y, window)
    
        X = np.zeros((len(X1), 3))
        X[:,0]=X1
        X[:,1]=X2
        X[:,2]=X3
        
        d['X'] = X
        d['Y'] = Y
        d['K'] = 3
        d['T'] = len(Y)
        d['pdf1'] = pdf1
        d['pdf2'] = pdf2
        d['pdf3'] = pdf3
        d['quant1'] = quant1
        d['quant2'] = quant2
        d['quant3'] = quant3
        scio.savemat(foldOut+f[:-4]+'_blurred_rate'+str(rate)+'_freq1_'+str(it)+'.mat', mdict=d)  
                    
exit


## IID model
# Sample rate 1
nSampAll = [50, 200, 800]#40 # Window size
DecimateAll = [200, 400, 100] # if the time series is subsampled set this to 0. If not, this sets the subsampling rate
#fold = 'StateDynamics3/'
#foldOut = 'Simulated1dData3/'
fold = 'C:/Users/Kevin Cheng/Box Sync/SimplexRandomWalk/TimeSeriesClustering/oneDimensional/data/SimplexTrajectories/'
foldOut = 'C:/Users/Kevin Cheng/Box Sync/SimplexRandomWalk/TimeSeriesClustering/oneDimensional/data/TimeSeries/'
try:
    os.mkdir(foldOut)
except:
    print('fold exists')

for nSamp in nSampAll:
    for Decimate in DecimateAll:
        for f in os.listdir(fold):
            Y = np.array([])
            d = scio.loadmat(fold+f)
            try:
                X = d['X']
            except:
                X = d['x']
                
            if (Decimate !=0):
                X = X[::Decimate,:]
                    
            (T,K) = np.shape(X)
            for i in range(T-1):
                quantOut = X[i,0]*quant1 + X[i,1]*quant2 + X[i,2]*quant3
                window = np.random.choice(quantOut, nSamp, True)
                Y = np.append(Y, window)
            d['Y'] = Y
            d['K'] = 3
            d['T'] = len(Y)
            d['pdf1'] = pdf1
            d['pdf2'] = pdf2
            d['pdf3'] = pdf3
            d['quant1'] = quant1
            d['quant2'] = quant2
            d['quant3'] = quant3
            scio.savemat(foldOut+f[:-4]+'_win'+str(nSamp)+'_stride'+str(Decimate)+'.mat', mdict=d)

## Perfect observation model
# Sample rate 1
nSampAll = [50]#40 # Window size
DecimateAll = [200] # if the time series is subsampled set this to 0. If not, this sets the subsampling rate
#fold = 'StateDynamics3/'
#foldOut = 'Simulated1dData3/'
fold = 'C:/Users/Kevin Cheng/Box Sync/SimplexRandomWalk/TimeSeriesClustering/oneDimensional/data/SimplexTrajectories/'
foldOut = 'C:/Users/Kevin Cheng/Box Sync/SimplexRandomWalk/TimeSeriesClustering/oneDimensional/data/TimeSeries/'
try:
    os.mkdir(foldOut)
except:
    print('fold exists')

for nSamp in nSampAll:
    for Decimate in DecimateAll:
        for f in os.listdir(fold):
            Y = np.array([])
            d = scio.loadmat(fold+f)
            try:
                X = d['X']
            except:
                X = d['x']

            if (Decimate !=0):
                X = X[::Decimate,:]
                    
            (T,K) = np.shape(X)
            for i in range(T-1):
                quantOut = X[i,0]*quant1 + X[i,1]*quant2 + X[i,2]*quant3
                Y = np.append(Y, quantOut[::int(nSupp/nSamp)])
            d['Y'] = Y
            d['K'] = 3
            d['T'] = len(Y)
            d['pdf1'] = pdf1
            d['pdf2'] = pdf2
            d['pdf3'] = pdf3
            d['quant1'] = quant1
            d['quant2'] = quant2
            d['quant3'] = quant3
            scio.savemat(foldOut+f[:-4]+'_perfect_win'+str(nSamp)+'_stride'+str(Decimate)+'.mat', mdict=d)    

