# -*- coding: utf-8 -*-
import sys

class TimeSeriesParams():
    def __init__(self, params={}):
        self.dParams={}
        # set default
        
         # Run Config Parameters
        self.dParams['dataSet'] = None
        self.dParams['logFile'] = None
        self.dParams['paramFile'] = 'params.txt'
        self.dParams['debugFolder'] = None
        self.dParams['outputFile'] = 'out.mat'
        self.dParams['dataName'] = 'test'
        self.dParams['dataFile'] = None
        self.dParams['dataVariable'] = 'y'
        self.dParams['initMethod'] = 'random'
        self.dParams['inputType'] = 'Sample'
        self.dParams['transitionModel'] = 'bimodal'
        self.dParams['initFile'] = None # Init file that holds all fixed parameters
        self.dParams['fixState'] = False
        self.dParams['fixPureState'] = False
        self.dParams['fixTransitionParams'] = False
        self.dParams['fixStateVariable'] = 'X'
        self.dParams['fixPureStateVariable'] = 'pureStates'
        self.dParams['fixTransitionParams'] = False
        self.dParams['fixTransitionAlphaVariable'] = None
        self.dParams['fixTransitionBetaVariable'] = None
#        self.dParams['distributionModel'] = 'PointCloud'
        self.dParams['distributionModel'] = 'FixedGrid'
                
         # Model Parameters
        self.dParams['alpha'] = 1.1
        self.dParams['beta'] = 10
        self.dParams['AB_weight'] = 0.5
        self.dParams['window'] = 1
        self.dParams['stride'] = 1
        self.dParams['offset'] = 0
        self.dParams['eps'] = 1e-4
        self.dParams['obsLossModel'] = 'Wass'
        self.dParams['pureStateLoss'] = 'MinWindowDist' #'None' #'WassGauss'
        self.dParams['regObs_Gauss'] = 100
        self.dParams['regTheta_Gauss'] = 1
        self.dParams['regGamma_Gauss'] = 1
        self.dParams['regObs_FG'] = 10000
        self.dParams['regTheta_FG'] = 100
        self.dParams['regGamma_FG'] = 1
        self.dParams['regObs_PC'] = 10000
        self.dParams['regTheta_PC'] = 100
        self.dParams['regGamma_PC'] = 1
        self.dParams['regObs_Q'] = 1
        self.dParams['regTheta_Q'] = 1
        self.dParams['regGamma_Q'] = 1
        self.dParams['normalizeData'] = False 
        
        # Point Cloud Parameters
        self.dParams['pcObsLoss'] = 'Wass' # 'Wass' or 'KDE'
        self.dParams['npcBary'] = 15 # number of barycenter points
        self.dParams['npcPure'] = 15 # number of barycenter points
        self.dParams['npcReference'] = 15 # number of barycenter points
        self.dParams['pcObsKdeBandwidth'] = 10.0 # standard deviation for isotropic Gaussian kernel used for KDE in comparing observations to barycenters
        self.dParams['pcPureStateBandwidth'] = 2.0 # standard deviation for isotropic Gaussian kernel used for KDE in comparing pure states to data distr.
        self.dParams['datLikelihoodDecimateFactor'] = 4 # This hopefully reducest he memory required 
        self.dParams['pcObsModel'] = 'regularized'

        # Grid Barycenter Params
        self.dParams['gridObsLoss'] = 'Wass' # 'Wass' or # 'Dir'
        self.dParams['nGrid'] = 10 # Grid size
        self.dParams['sinkhornE'] = 2.0 #0.40# 15.0 #0.5 # Regularzation parameter for sinkhorn computation
        self.dParams['obsDirichletAlpha'] = 1.01 # Sum of dirichlet prior on observation
        self.dParams['gridKdeBandwidth'] = 2.0 # Gaussian kernel bandwidth of Grid KDE estimation
        self.dParams['gridObsModel'] = 'regularized'
        self.dParams['gridLimits'] = None
        self.dParams['attachIter'] = 3
        self.dParams['detachIter'] = 0
        
        # Gaussian Model Specific Parameters
        self.dParams['geoMean'] = "Euc" # Geometry to evaluate and optimize the Mean
        self.dParams['geoCov'] = "Wass" # Geometry to evaluate and optimize the Covariance
        self.dParams['gaussObsModel'] = 'regularized'
        self.dParams['cluster_sig'] = 100 # Prior on Wasserstein distance to Cluster center: Gaussian standard deviation 
        
        # Time series dynamics Dirichlet parameters
        self.dParams['dirichletVar'] = 5.0

        # Optimization Parameters
        self.dParams['nOptimStep'] = 10000 #50000 
        self.dParams['lr_Gamma_Gauss'] = 2e-3 #2e-3 # Used for the Gaussian formulation
        self.dParams['lr_Gamma_FG'] = 1e-2 
        self.dParams['lr_Gamma_PC'] = 1e-2 
        self.dParams['lr_Gamma_Q'] = 1e-2 
        self.dParams['lr_AB'] = 2e-3 #2e-3 # For Gaussian Formualation
        self.dParams['lr_Cluster_Gauss'] = 1e-3 # 1e-3 # For Gaussian Formulation
        self.dParams['lr_Cluster_FG'] = 1e-2 # 1e-3 # For FG Formulation
        self.dParams['lr_Cluster_PC'] = 1e-1 # 1e-3 # For PC Formulation
        self.dParams['lr_Cluster_Q'] = 1e-4 # 1e-3 # For PC Formulation
        self.dParams['nCyclic'] = 200
        self.dParams['printInterval'] = 50
        self.dParams['cyclicThresh'] = 1
        self.dParams['cyclicIterMax'] = 1000
        self.dParams['optimMethod_Cluster'] = "LineSearch"
        self.dParams['cyclic_MuSig'] = True
        self.dParams['sgdBatch'] = 0 # stochastic gradient descent batch size. If 0, then use full dataset.
        
        # Internal Parameters
        self.dParams['T'] = 0
        self.dParams['dim'] = 0
        self.dParams['K'] = 0
        
        # Update and Save
        self.update(params)
        self.save()
        
    def save(self):
        # Store parameters"
         # Run Config Parameters
        self.dataSet = self.dParams['dataSet']
        self.logFile = self.dParams['logFile']
        self.paramFile = self.dParams['paramFile'] 
        self.debugFolder = self.dParams['debugFolder']
        self.outputFile = self.dParams['outputFile']
        self.dataName = self.dParams['dataName']
        self.dataFile = self.dParams['dataFile']
        self.dataVariable = self.dParams['dataVariable']
        self.initMethod = self.dParams['initMethod']
        assert (self.initMethod == 'CPD' or self.initMethod == 'GMM' or self.initMethod == 'label' or self.initMethod == 'precomputed' or self.initMethod == 'random')
        self.inputType = self.dParams['inputType']
        assert (self.inputType== 'Sample' or self.inputType == 'Gaussian')
        self.transitionModel = self.dParams['transitionModel']
        assert (self.transitionModel== 'unimodal' or self.transitionModel == 'bimodal')
        self.initFile = self.dParams['initFile']
        self.fixState = self.dParams['fixState']
        self.fixPureState = self.dParams['fixPureState']
        self.fixTransitionParams = self.dParams['fixTransitionParams']
        self.fixStateVariable = self.dParams['fixStateVariable']
        self.fixPureStateVariable = self.dParams['fixPureStateVariable']
        self.fixTransitionParams = self.dParams['fixTransitionParams']
        self.fixTransitionAlphaVariable = self.dParams['fixTransitionAlphaVariable'] 
        self.fixTransitionBetaVariable = self.dParams['fixTransitionBetaVariable']
        self.distributionModel = self.dParams['distributionModel']
        assert (self.distributionModel== 'Gauss' or self.distributionModel == 'FixedGrid' or self.distributionModel == 'PointCloud' or self.distributionModel == 'Quantile')

         # Model Parameters
        self.alpha = self.dParams['alpha']
        self.beta = self.dParams['beta']
        self.AB_weight = self.dParams['AB_weight']
        self.window = self.dParams['window']
        self.stride = self.dParams['stride']
        self.offset = self.dParams['offset']
        self.eps = self.dParams['eps']
        self.obsLossModel = self.dParams['obsLossModel']
        assert (self.obsLossModel== 'Wass' or self.obsLossModel == 'likelihood' or self.obsLossModel == 'Dirichlet')
        self.pureStateLoss = self.dParams['pureStateLoss']
        assert (self.pureStateLoss== 'Wass' or self.pureStateLoss=='WassGauss' or self.pureStateLoss == 'likelihood' or self.pureStateLoss == 'KL' or self.pureStateLoss == 'MinWindowDist' or self.pureStateLoss == 'None')
        self.regObs_Gauss = self.dParams['regObs_Gauss']
        self.regTheta_Gauss = self.dParams['regTheta_Gauss']
        self.regGamma_Gauss = self.dParams['regGamma_Gauss']
        self.regObs_FG = self.dParams['regObs_FG']
        self.regTheta_FG = self.dParams['regTheta_FG']
        self.regGamma_FG = self.dParams['regGamma_FG']
        self.regObs_PC = self.dParams['regObs_PC']
        self.regTheta_PC = self.dParams['regTheta_PC']
        self.regGamma_PC = self.dParams['regGamma_PC']
        self.regObs_Q = self.dParams['regObs_Q']
        self.regTheta_Q = self.dParams['regTheta_Q']
        self.regGamma_Q = self.dParams['regGamma_Q']
        self.normalizeData = self.dParams['normalizeData']

        # Point Cloud Parameters
        self.pcObsLoss = self.dParams['pcObsLoss']
        self.npcBary = self.dParams['npcBary']
        self.npcReference = self.dParams['npcReference']
        self.npcPure = self.dParams['npcPure']
        self.pcObsKdeBandwidth = self.dParams['pcObsKdeBandwidth']
        self.pcPureStateBandwidth = self.dParams['pcPureStateBandwidth']
        self.datLikelihoodDecimateFactor = self.dParams['datLikelihoodDecimateFactor']
        self.pcObsModel = self.dParams['pcObsModel']

        # Grid Barycenter Params
        self.gridObsLoss = self.dParams['gridObsLoss']
        self.nGrid = self.dParams['nGrid']
        self.sinkhornE = self.dParams['sinkhornE']
        self.obsDirichletAlpha = self.dParams['obsDirichletAlpha']
        self.gridKdeBandwidth = self.dParams['gridKdeBandwidth']
        self.gridObsModel = self.dParams['gridObsModel']
        self.gridLimits = self.dParams['gridLimits']
        self.attachIter = self.dParams['attachIter']
        self.detachIter = self.dParams['detachIter']
        
        # Gaussian Model Specific Parameters
        self.geoMean = self.dParams['geoMean'] 
        assert (self.geoMean == 'Euc')
        self.geoCov = self.dParams['geoCov']
        assert (self.geoCov == 'Wass' or self.geoCov == 'Euc' or self.geoCov == 'Hel' or self.geoCov == 'GMM')
        self.gaussObsModel = self.dParams['gaussObsModel']
        assert (self.gaussObsModel == 'regularized' or self.gaussObsModel == 'likelihood')
        self.cluster_sig = self.dParams['cluster_sig']
        
        # Time series Dirichlet parameters
        self.dirichletVar = self.dParams['dirichletVar']

         # Optimization Parameters
        self.nOptimStep = self.dParams['nOptimStep']
        self.lr_Gamma_Gauss = self.dParams['lr_Gamma_Gauss']
        self.lr_Gamma_FG = self.dParams['lr_Gamma_FG']
        self.lr_Gamma_PC = self.dParams['lr_Gamma_PC']
        self.lr_Gamma_Q = self.dParams['lr_Gamma_Q']
        self.lr_AB = self.dParams['lr_AB']
        self.lr_Cluster_Gauss = self.dParams['lr_Cluster_Gauss']
        self.lr_Cluster_PC = self.dParams['lr_Cluster_PC']
        self.lr_Cluster_FG = self.dParams['lr_Cluster_FG']
        self.lr_Cluster_Q = self.dParams['lr_Cluster_Q']
        self.nCyclic = self.dParams['nCyclic']
        self.printInterval = self.dParams['printInterval']
        self.cyclicThresh = self.dParams['cyclicThresh']
        self.cyclicIterMax = self.dParams['cyclicIterMax']
        self.sgdBatch = self.dParams['sgdBatch']

        self.optimMethod_Cluster = self.dParams['optimMethod_Cluster']
        assert (self.optimMethod_Cluster == 'GradientDescent' or self.optimMethod_Cluster == 'LineSearch')
        self.cyclic_MuSig = self.dParams['cyclic_MuSig']

        
         # Internal Parameters
        self.T = self.dParams['T']
        self.dim = self.dParams['dim']
        self.K = self.dParams['K']

    def update(self, params):
        # Update based on inputs
        for k in params.keys():
            if (params[k] is not None):
                self.dParams[k] = params[k]
        self.save()

        
    def write(self, f=sys.stdout):
        if (f is None):
            f = sys.stdout
        else:
            f = open(f, 'w')
        print("Parameter Dump", file=f)
        for k in self.dParams.keys():
            print("  " + k + ": \t" + str(self.dParams[k]), file = f)
        print(" ", file=f)

        if (f != sys.stdout):
            f.close()
