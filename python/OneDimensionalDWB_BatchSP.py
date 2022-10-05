import sys
import TimeSeriesParams as TSP
import numpy as np
import scipy.io as scio
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint
import scipy.sparse as scisparse
import os, ot, sys
from sklearn.cluster import SpectralClustering

def WindowData(datO, window, stride, offset=0):
    halfWin = int(np.floor(window/2))
    if (len(datO.shape) == 1):
        dat = np.expand_dims(datO,axis=1)
    
    out = []
    for i in range(offset, len(datO)-halfWin, stride):
        if (i-halfWin >= 0 and i-halfWin+window <= len(datO)):
            out.append(datO[i-halfWin:i-halfWin+window,:])
        
    return np.asarray(out)

def Barycenter(x,pS):
    return np.sum(np.diag(x) @ pS, axis = 0)

def ClusterInit(Y,K):
    (T, n) = np.shape(Y)
    dist = np.zeros((T,T))
    for i in range(T):
        for j in range(T):
            if (i!= j):
                dist[i,j] = 1/n*np.linalg.norm(Y[i]-Y[j])**2
    clustering = SpectralClustering(n_clusters=K,affinity='precomputed').fit(np.exp(-dist))
    Q=np.zeros((K,n))
    for i in range(K):
        ind = np.argwhere(clustering.labels_==i)
        Q[i] = np.mean(Y[ind],axis=0)
    return Q
                

class Scipy_1dEmpiricalDistribution():
    regX = 1
    regQ = 1
    T = 0 
    D = 0
    K = 0
    eps = 1e-3
    aCosEps = 1e-4
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
        (self.T, self.D) = np.shape(Y)
        (dump, self.K) = np.shape(X)
        self.Y = Y.flatten()
        self.X = X.flatten()
        self.Q = Q.flatten()
        
#        self.LinCon = LinearConstraint(self.X, self.eps*np.ones((self.T, self.K)), (1-self.eps)*np.ones((self.T,self.K)))
        
        self.consX=[]
        self.consX.append({'type': 'eq', 'fun': lambda x:  np.sum(np.reshape(x,(self.T,self.K)), axis=1)-1})
#        self.consX.append({'type': 'eq', 'fun': lambda x:  x[0::3]+x[1::3]+x[2::3]-1})
        self.consX.append({'type': 'ineq', 'fun': lambda x:  x-self.eps})
        self.consX.append({'type': 'ineq', 'fun': lambda x:  (1-self.eps)-x})

        self.bndsX=[]

        self.consQ=[]
        self.consQ.append({'type': 'ineq', 'fun': lambda q:  (np.reshape(q, (self.K, self.D))[:,1:] - np.reshape(q, (self.K, self.D))[:,:-1]).flatten()})
#        self.consQ.append({'type': 'ineq', 'fun': lambda q:  q[1:self.D] - q[0:self.D-1]})
#        self.consQ.append({'type': 'ineq', 'fun': lambda q:  q[self.D+1:2*self.D] - q[self.D:2*self.D-1]})
#        self.consQ.append({'type': 'ineq', 'fun': lambda q:  q[2*self.D+1:3*self.D] - q[2*self.D:3*self.D-1]})
        self.bndsQ=[]
#        self.consQ = ThisSucks1000()
        self.mode = mode


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
        Q2 = np.reshape(Q, (self.K,self.D))
        return self.T*1/self.D*np.sum( (( np.eye(self.K) - 1/self.K *np.ones((self.K,self.K)))@Q2)**2)
    
    def CostFunctionData(self,Q,X,Y):
        X2 = np.reshape(X, (self.T,self.K))
        Q2 = np.reshape(Q, (self.K,self.D))
        Y2 = np.reshape(Y, (self.T,self.D))   
        dataLoss = np.sum(( Y2- X2 @ Q2 )**2)
        return 1/self.D* ( dataLoss )
    
    def CostFuncData_JacQ(self, Q,X,Y):
        X2 = np.reshape(X, (self.T,self.K))
        Q2 = np.reshape(Q, (self.K,self.D))
        Y2 = np.reshape(Y, (self.T,self.D))
        tmp = -2/self.D*np.transpose(X2) @ ( Y2 - X2 @ Q2 )
        return tmp.flatten()
        
    def CostFuncData_JacX(self, Q,X,Y):
        X2 = np.reshape(X, (self.T,self.K))
        Q2 = np.reshape(Q, (self.K,self.D))
        Y2 = np.reshape(Y, (self.T,self.D)) 
        tmp = -2/self.D * ( Y2 - X2 @ Q2 ) @ np.transpose(Q2)
        return tmp.flatten()
    
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
        res = minimize(self.CostOptimizeQ, self.Q, args=(X.flatten(), self.Y), 
                       method='SLSQP', constraints=self.consQ, tol = 1e-5)#, jac=self.CostFuncData_JacQ)
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
    QInit = sys.argv[3] # 'GT' or 'random
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
    nIter = 60
#    stride = window# this should always be the same
    convergence = 1e-8
    
    # Print input parameters    
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
    if (stride==0): # Dsjoint windows
        Y = np.sort(np.squeeze(WindowData(dat, window, window, offset=int(np.floor(window/2)))), axis=1)
    else:
        Y = np.sort(np.squeeze(WindowData(dat, window, stride, offset)), axis=1)
        
    (p.T,dump) = np.shape(Y)
    
    # Setup init
    QO = np.zeros((p.K,window))
    if (sim==1):
        QGT_O = np.zeros((p.K,100000))
        QGT = np.zeros((p.K,window))
        tmp = np.linspace(0,100000-1,window).astype(int)
        QGT_O[0] = d['quant1'].flatten()
        QGT_O[1] = d['quant2'].flatten()
        QGT_O[2] = d['quant3'].flatten()
    
        QGT[0] = d['quant1'].flatten()[tmp]
        QGT[1] = d['quant2'].flatten()[tmp]
        QGT[2] = d['quant3'].flatten()[tmp]
        XGT_O = d['X']
        halfWin = int(window/2)
        if (stride == 0):
            XGT = XGT_O[np.linspace(halfWin, halfWin+window*np.floor(p.T-1), p.T).astype(int),:]
        else:
            XGT = np.zeros((p.T,p.K))
            count = 0
            for i in range(offset, len(XGT_O)-halfWin, stride):
                if (i-halfWin >= 0 and i-halfWin+window <= len(XGT_O)):
                    XGT[count,:] = XGT_O[i,:]
                    count = count + 1
            #        XGT = XGT[np.linspace(0,len(XGT)-1, p.T).astype(int),:]

    nInit = 1
    if (QInit == 'random'):
        nInit = 1001

    # For Random initialization
    np.random.seed(seed)
    randInitIdx = np.zeros((nInit, p.K)).astype(int)
    for i in range(nInit):
        randInitIdx[i] = np.random.choice(range(p.T), p.K, False)
#    randInitIdx[0,:] = [10, 80, 140]

    doPlot = False

    res = np.zeros(nInit)
    resData = np.zeros(nInit)
    resQ = np.zeros(nInit)
    resX = np.zeros(nInit)
    resHistQ = np.zeros(nInit)
    resHistX = np.zeros(nInit)


    gtResDictGT={}
    gtResDictClust={}
    saveDict={}
    gtDict={}
    
    # Start optimization
    bestSoFar=0
    XOut = np.zeros((p.T, p.K, nInit))
    QOut = np.zeros((p.K, window, nInit))

    for it in range(nInit):

        print("Run file " + inputFile + "  run# " + str(it))
        Q=QO.copy()         
        hist = []
        histData = []
        histXReg =[]
        histQReg =[]
        histQ = []
        histX = []
        if (QInit == 'cluster' or (QInit == 'random' and it ==0)):
            print("Cluster Init")
            Y2 = np.sort(np.squeeze(WindowData(dat, clustInitWin, clustInitWin, offset = int(np.floor(clustInitWin/2)))), axis=1)
            Qtmp = ClusterInit(Y2, p.K)
            idx = np.linspace(0, clustInitWin-1, window).astype(int)
            Q = Qtmp[:,idx]
            QO = Q.copy()
        elif(QInit=='GT'): # start with ground turth init
            Q=QGT.copy()
            QO=Q.copy()
        elif(QInit =='file'):
            initD = scio.loadmat(initFile)
            Qtmp = initD['Q']
            (dump, fileInitN) = np.shape(Qtmp)
            idx = np.linspace(0, fileInitN-1, window).astype(int)
            Q = Qtmp[:,idx]
            QO = Q.copy()
        elif (QInit=='random'):
            print("Random Init")
            for i in range(p.K):
                QO[i] = Y[randInitIdx[it,i]]
            Q=QO.copy()
            
        
        X = np.ones((p.T, p.K))/p.K
#        X = np.random.rand(p.T, p.K)
#        X = np.transpose(np.transpose(X)/np.sum(X,axis=1))

        spOpt = Scipy_1dEmpiricalDistribution(Y, Q, X, regX=regX, regQ=regQ, mode = distModel)
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
        if (sim==1):
            XOut[:,:,it] = X@T
            QOut[:,:,it] = T@Q
        else:
            XOut[:,:,it] = X
            QOut[:,:,it] = Q
        res[it] = hist[-1]
        resData[it] = histData[-1]
        resHistQ[it] = histQReg[-1]
        resHistX[it] = histXReg[-1]
        if (sim==1):
            resQ[it] = histQ[-1]
            resX[it] = histX[-1]
            
        if (QInit == 'GT'):
            gtResDictGT={'X_GT_init':X, 'dat':dat, 'Y':Y, 'QO_GT_init':QO, 'Q_GT_init':Q}
        elif (QInit == 'cluster' or QInit == 'file'):
            gtResDictClust={'X_clust_init':X, 'dat':dat, 'Y':Y, 'QO_clust_init':QO, 'Q_clust_init':Q}
        elif (QInit == 'random' and hist[-1] < bestSoFar):
            bestSoFar = hist[-1]
            saveDict={'X':X@T, 'dat':dat, 'Y':Y, 'QO':T@QO, 'Q':T@Q, 
                      'histData':histData, 'histQ':histQ, 'hist':hist, 'histX':histX, 
                      'initIdx':it, 'T':T, 'QInit':QInit, 'bestSoFar':bestSoFar}
        if (sim==1):
            gtDict = {'XGT':XGT, 'QGT':QGT, 'QGT_O':QGT_O}
            
#        scio.savemat(outputFile, mdict={**gtResDictGT, **gtResDictClust, **saveDict, 'XGT':XGT, 'dat':dat, 'Y':Y, 
        scio.savemat(outputFile, mdict={**gtResDictGT, **gtResDictClust, **saveDict, **gtDict,
                                'window':window, 'regX':regX, 'regQ':regQ,
                                'randInitIdx':randInitIdx, 'XOut':XOut, 'QOut':QOut,
                                'res':res, 'resData':resData, 'resQ':resQ, 'resX':resX, 'resHistQ':resHistQ, 'resHistX':resHistX})
