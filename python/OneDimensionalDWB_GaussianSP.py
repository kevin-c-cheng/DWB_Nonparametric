import sys
import numpy as np
import scipy.io as scio
import TimeSeriesParams as TSP
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint
import scipy.sparse as scisparse
import os, ot, sys
from sklearn.cluster import SpectralClustering
#from ThisSucks import ThisSucks1000

def WindowData(datO, window, stride, offset=0):
    halfWin = int(np.floor(window/2))
    if (len(datO.shape) == 1):
        dat = np.expand_dims(datO,axis=1)
    
    out = []
    for i in range(offset, len(datO)-offset, stride):
        if (i-halfWin >= 0 and i-halfWin+window <= len(datO)):
            out.append(datO[i-halfWin:i-halfWin+window,:])
        
    return np.asarray(out)

def Barycenter(x,pS):
    return np.sum(np.diag(x) @ pS, axis = 0)

def ClusterInit_Gauss(Y,K):
    (T, n) = np.shape(Y)
    dist = np.zeros((T,T))
    for i in range(T):
        for j in range(T):
            if (i!= j):
                dist[i,j] = GaussWassDist(Y[i], Y[j])
    clustering = SpectralClustering(n_clusters=K,affinity='precomputed').fit(np.exp(-dist))
    Q=np.zeros((K,2))
    for i in range(K):
        ind = np.squeeze(np.argwhere(clustering.labels_==i))
        Q[i] = GaussWassBary(1/len(ind)*np.ones((1,len(ind))), Y[ind])
    return Q

def GaussWassBary(X, Q):
    if (len(np.shape(Q))==1):
        Q = np.expand_dims(Q,0)
    Qm = Q[:,0]
    QS = Q[:,1]
    tmp = (X@np.sqrt(QS))**2
    return np.transpose(np.concatenate((np.expand_dims(X@Qm,0), np.expand_dims(tmp,0))))

def GaussWassDist(G1, G2):
    if (len(np.shape(G1))==1):
        G1 = np.expand_dims(G1,0)
        G2 = np.expand_dims(G2,0)
        
    distMean = np.sum((G1[:,0] - G2[:,0])**2) # Euclidean distance betweeen means
    distCov = np.sum(G1[:,1] + G2[:,1] - 2*np.sqrt(G1[:,1] * G2[:,1]))
    return distMean+distCov # Square Wasserstein distance

def EstimateGaussian(Y0):
    (T,n) = np.shape(Y0)
    Y = np.zeros((T,2))
    for i in range(T):
        Y[i,0] = np.mean(Y0[i,:])
        Y[i,1] = np.var(Y0[i,:], ddof=1)
    return Y                
    
class Scipy_Gauss():
    regX = 1
    regQ = 1
    T = 0 
    D = 0
    K = 0
    eps = 1e-3
    aCosEps = 1e-4
    minGaussVar = 1e-3
    consX=[]
    bndsX=[]
    consQ=[]
    bndsQ=[]
    LinCon=None
    
    Y = []
    X = []
    Q = []
    mode = 'Euclidean'
    
    def __init__(self, Y, Q, X, regX = 1.0, regQ=1.0, mode = 'Euclidean'):
        self.regX = regX
        self.regQ = regQ
        self.T = len(Y)
        self.D = 2
        (dump, self.K) = np.shape(X)
        self.Y = Y.flatten()
        self.X = X.flatten()
        self.Q = Q.flatten()
        
#        self.LinCon = LinearConstraint(self.X, self.eps*np.ones((self.T, self.K)), (1-self.eps)*np.ones((self.T,self.K)))
        
        self.consX=[]
        self.consX.append({'type': 'eq', 'fun': lambda x:  np.sum(np.reshape(x,(self.T,self.K)), axis=1)-1})
        self.consX.append({'type': 'ineq', 'fun': lambda x:  x-self.eps})
        self.consX.append({'type': 'ineq', 'fun': lambda x:  (1-self.eps)-x})
        self.bndsX=[]

        self.consQ=[]
        self.consQ.append({'type': 'ineq', 'fun': lambda x:  x[1::2]-self.minGaussVar})
        self.bndsQ=[]
#        self.bndsQ=((None,None), (self.minGaussVar, None))*self.K

        self.mode = mode
        #Gaussians are parameterized by their mean and variance
        

    def CostFunction(self,X,Q,Y):            
        dataLoss = self.CostFunctionData(Q,X,Y)
        XLoss = self.CostFunctionX(X)
        QLoss = self.CostFunctionQ(Q)
        return (dataLoss) + self.regX*XLoss + self.regQ*QLoss

    def CostFunctionX(self,X):
        X2 = np.reshape(X, (self.T,self.K))
        if (self.mode == "Euclidean"):
            stateLoss = np.sum((X2[1:]- X2[0:-1])**2)
        elif(self.mode == "BhatACos"):
            tmp = np.sum(np.sqrt(X2[1:])*np.sqrt(X2[0:-1]),1)
            stateLoss = np.sum(np.arccos(np.clip(tmp, self.aCosEps, 1-self.aCosEps)))
        elif(self.mode == "MatusitaSq"):
            stateLoss = np.sum((np.sqrt(X2[1:]) - np.sqrt(X2[0:-1]))**2)
        elif(self.mode == 'Aitchison'):
            gX = np.expand_dims(np.power(np.prod(X2[1:], 1),self.K), 1) @ np.ones((1,self.K))
            gXp1 = np.expand_dims(np.power(np.prod(X2[0:-1], 1),self.K), 1) @ np.ones((1,self.K))
            stateLoss = np.sum( (np.log(X2[1:]/gX) - np.log(X2[0:-1]/gXp1))**2)
        return stateLoss

    def CostFunctionQ(self,Q):
        Q2 = np.reshape(Q, (self.K,2))
        bary = GaussWassBary(1/p.K*np.ones((1,p.K)), Q2)
        return self.T*GaussWassDist(Q2, bary)
    
    def CostFunctionData(self,Q,X,Y):
        X2 = np.reshape(X, (self.T,self.K))
        Q2 = np.reshape(Q, (self.K,self.D))
        Y2 = np.reshape(Y, (self.T,self.D))
        
        baryT = GaussWassBary(X2, Q2)
        return 1/self.D* GaussWassDist(Y2, baryT)
        
    def CostOptimizeX(self, X, Q,Y):
        dataLoss = self.CostFunctionData(Q,X,Y)
        XLoss = self.CostFunctionX(X)
        return ( dataLoss ) + self.regX*XLoss

    def CostOptimizeQ(self, Q, X,Y):
        dataLoss = self.CostFunctionData(Q,X,Y)
        QLoss = self.CostFunctionQ(Q)
        return ( dataLoss ) + self.regQ*QLoss
    

    def OptimizeX(self,Q):
        res = minimize(self.CostOptimizeX, self.X, args=(Q.flatten(), self.Y), 
                   method='SLSQP', constraints=self.consX, tol = 1e-5, bounds=self.bndsX)
#                   method='SLSQP', constraints=(self.LinCon,self.consX), tol = 1e-5, bounds=self.bndsX)
        self.X = res.x
        return np.reshape(res.x, (self.T, self.K))
    
    def OptimizeQ(self,X):
        inf=99999
        res = minimize(self.CostOptimizeQ, self.Q, args=(X.flatten(), self.Y), 
                       method='SLSQP', constraints=self.consQ, tol = 1e-5, bounds=self.bndsQ)#, jac=self.CostFuncData_JacQ)
        res.x = np.maximum(res.x, [-inf, self.minGaussVar]*self.K)
        self.Q = res.x
        return np.reshape(res.x, (self.K, self.D))

def GetPermutation(Q, QGT):
    (n,d) = Q.shape
    (nF,dF) = QGT.shape
    dup = int(dF/d)
    
    cost=np.zeros((n,n))
    for i in range(n):
        tmp = []
        for k in range(dup):
            tmp.append(Q[i])
        tmp=np.sort(np.stack(tmp).flatten())
        for j in range(n):
            cost[i,j] = 1/dF*np.linalg.norm(tmp-QGT[j])**2
    a = np.ones(n)
    T = ot.emd(a,a,cost)
    return T

def EvaluationDataGauss(Y, Q, X, nSamp=10000):
    bary = GaussWassBary(X,Q)
    (T, n) = np.shape(Y)
    dist = np.zeros(T)
    for i in range(T):
        samps = np.random.normal(bary[i,0], np.sqrt(bary[i,1]), nSamp )
        a = 1/nSamp*np.ones(nSamp)
        b = 1/n*np.ones(n)
        M = ot.dist(np.expand_dims(samps,1), np.expand_dims(Y[i],1), metric='sqeuclidean')
        C = ot.emd(a,b,M)
        dist[i] = np.sum(M*C)
    return np.sum(dist)
        

def EvaluationQ(Q, QGT):
    # assumes Q and QGT are the already matched
    (n,d) = Q.shape
    (nF,dF) = QGT.shape
    dup = int(dF/d)
    lcm = np.lcm(dF, d)
    
    dist = np.zeros(n)
    for i in range(n):
        Q_tmp = np.repeat(Q[i], int(lcm/d))
        QGT_tmp = np.repeat(QGT[i], int(lcm/dF))
        dist[i] = 1/lcm*np.linalg.norm(Q_tmp-QGT_tmp)**2
    return np.sum(dist)

def EvaluationX(X, XGT):
    # assumes X and XGT are the already matched
    return np.sum(np.linalg.norm(X-XGT)**2)


if __name__=="__main__":
    # Input parameters
    inputFile = sys.argv[1]
    outputFile = sys.argv[2]
    QInit = sys.argv[3] # 'cluster' or 'file'
    seed = int(sys.argv[4])
    distModel = sys.argv[5] # 'BhatACos', 'MatusitaSq'
    window = int(sys.argv[6])
    regX = float(sys.argv[7])
    regQ = float(sys.argv[8])
    stride = int(sys.argv[9])
    offset = int(sys.argv[10])
    clustInitWin = int(sys.argv[11])
    sim = int(sys.argv[12])
    initFile = sys.argv[13]
    nIter = 500
#    stride = window# this should always be the same
    convergence = 1e-10
    
    # Print input parameters    
    print("InputFile: ", inputFile)
    print("outputFile: ", outputFile)
    print("QInit: ", QInit)
    print("seed: ", str(seed))
    print("distModel: ", distModel)
    print("window: ", str(window))
    print("regX: ", str(regX))
    print("regQ: ", str(regQ))
    print("stride: ", str(stride))
    print("offset: ", str(offset))
    print("clustInitWin: ", str(clustInitWin))
    print("sim ", str(sim))
    print("initFile ", str(initFile))
        
    p = TSP.TimeSeriesParams()
    fInd = 0

    # File specific parameters
    d = scio.loadmat(inputFile)
    p.K = d['K'].flatten()[0]        
    dat = d['Y'].transpose()
    if (stride==0): # Disjoint windows
        Y0 = np.sort(np.squeeze(WindowData(dat, window, window, offset=int(np.floor(window/2)))), axis=1)
    else:
        Y0 = np.sort(np.squeeze(WindowData(dat, window, stride, offset)), axis=1)
    Y = EstimateGaussian(Y0)
        
    (p.T,dump) = np.shape(Y)
    
    # Setup init
    QO = np.zeros((p.K,window))

    if (sim==1): # Simulated data we have ground truth    
        QGT_O = np.zeros((p.K,100000))
        QGT = np.zeros((p.K,window))
        tmp = np.linspace(0,100000-1,window).astype(int)
        QGT_O[0] = d['quant1'].flatten()
        QGT_O[1] = d['quant2'].flatten()
        QGT_O[2] = d['quant3'].flatten()
    
        QGT[0] = d['quant1'].flatten()[tmp]
        QGT[1] = d['quant2'].flatten()[tmp]
        QGT[2] = d['quant3'].flatten()[tmp]
        XGT = d['X']
        XGT = XGT[np.linspace(0,len(XGT)-1, p.T).astype(int),:]

    nInit = 1

    doPlot = False

    res = np.zeros(nInit)
    resData = np.zeros(nInit)
    resQ = np.zeros(nInit)
    resX = np.zeros(nInit)
    resHistQ = np.zeros(nInit)
    resHistX = np.zeros(nInit)


    gtResDictClust={}
    gtDict={}
    
    # Start optimization
    bestSoFar=0
    for it in range(nInit):
        print("Run file " + inputFile)
        hist = []
        histData = []
        histXReg =[]
        histQReg =[]
        histQ = []
        histX = []
        
        if(QInit == 'cluster'): # Clustering Init
            Y2 = np.sort(np.squeeze(WindowData(dat, clustInitWin, clustInitWin, offset = int(np.floor(clustInitWin/2)))), axis=1)
            Y3 = EstimateGaussian(Y2)
            Q = ClusterInit_Gauss(Y3, p.K)
            QO = Q.copy()
        elif (QInit=='file'):
            initD = scio.loadmat(initFile)
            Q = initD['QGauss']
            QO=Q.copy()
            
        
        X = np.ones((p.T, p.K))/p.K
#        X = np.random.rand(p.T, p.K)
#        X = np.transpose(np.transpose(X)/np.sum(X,axis=1))

        spOpt = Scipy_Gauss(Y, Q, X, regX=regX, regQ=regQ, mode = distModel)
        print('initLoss: ', spOpt.CostFunction(X,Q,Y))

        i = 0
        prevLossi = lossi = 0
        while (i < nIter ):#and (i<=1 or prevLossi-lossi > convergence)):
            prevLossi = lossi
            if (i%2 ==0): # Optimize for X
                X = spOpt.OptimizeX(Q)
            else: # Optimize for Q
                Q = spOpt.OptimizeQ(X)
            print("  i: " + str(i) + " loss: ", spOpt.CostFunction(X,Q,Y))
            lossi = spOpt.CostFunction(X,Q,Y)
            hist.append(spOpt.CostFunction(X,Q,Y))
            histData.append(spOpt.CostFunctionData(Q,X,Y))
            histXReg.append(spOpt.CostFunctionX(X))
            histQReg.append(spOpt.CostFunctionQ(Q))
            
            if (sim==1):
                T = GetPermutation(Q, QGT)
                histQ.append(EvaluationQ(T@Q, QGT_O)/ p.K)
                histX.append(EvaluationX(X@T, XGT)/p.T)
#            scio.savemat(outputFile, mdict={'XGT':XGT, 'dat':dat, 'Y':Y, 
#                                'QGT':QGT, 'QGT_O':QGT_O, 'window':window,
#                                'randInitIdx':randInitIdx,
#                                'res':res, 'resData':resData, 'resQ':resQ, 'resX':resX,
#                                'hist':hist, 'histData':histData, 'histQ':histQ, 'histX':histX,
#                                'Xtmp':X, 'Qtmp':Q, 'histXReg':histXReg, 'histQReg':histQReg})
            i=i+1


        res[it] = hist[-1]
        resData[it] = EvaluationDataGauss(Y0, Q, X) #histData[-1]
        resHistQ[it] = histQReg[-1]
        resHistX[it] = histXReg[-1]
        if (sim==1):
            resQ[it] = histQ[-1]
            resX[it] = histX[-1]
            
        if (it == 0):
            gtResDictClust={'X_clust_init':X, 'dat':dat, 'Y':Y, 'QO_clust_init':QO, 'Q_clust_init':Q}
            
        if (sim==1):
            gtDict = {'XGT':XGT, 'QGT':QGT, 'QGT_O':QGT_O}
        scio.savemat(outputFile, mdict={**gtResDictClust, **gtDict, 'window':window, 'regX':regX, 'regQ':regQ,
                                'res':res, 'resData':resData, 'resQ':resQ, 'resX':resX, 'resHistQ':resHistQ, 'resHistX':resHistX})
