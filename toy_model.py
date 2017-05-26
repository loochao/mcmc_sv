# -*- coding: utf-8 -*-
"""
Created on Fri May 26 02:55:27 2017

@author: Jiacheng Z
"""

from __future__ import print_function
import pandas as pd
import numpy as np
import numpy.random as rand
from datetime import datetime

def NOW():
    return str(datetime.now())[:-7]

class ReadData:
    def __init__(self,SplitYear=2013):
        file_loc = r"https://raw.githubusercontent.com/jiacheng0409/mcmc_sv/master/sp_daily.csv"
        rwData = pd.read_csv(file_loc)
        train_IDX = rwData['caldt'] > SplitYear*(10**4)
        self.train = rwData[train_IDX]
        self.test = rwData[~train_IDX]
        print('{0}\n[INFO] Finished data importing.'.format('='*20+NOW()+'='*20))

class PriorParameters:
    def __init__(self, TrainSerie,Seed = rand.randint(1)):
        rand.seed(Seed)
        TrainLen = len(TrainSerie)

        Mu = rand.randn()

        def AlphaPrior():
            MeanVec = rand.rand(2)
            CovMat = np.abs(rand.rand(2, 2))
            Alpha = rand.multivariate_normal(mean=MeanVec,cov=CovMat)
            return Alpha
        Alpha = AlphaPrior()
        # this abs(Alpha_2) <= 1 constraint make sure our AR(1) for volatility is stationary
        while np.abs(Alpha[1]>=1):  Alpha = AlphaPrior()

        def SigmaPrior():
            Lambda = rand.randn()
            m = rand.randint(low=1,high=10)
            DegreeOfFreedom = TrainLen + m -1
            sigma_sq_inv = rand.chisquare(DegreeOfFreedom)
            sigma_sq = float(m * Lambda) / sigma_sq_inv
            return sigma_sq
        Sigma_Sq = SigmaPrior()

        Epsilon_vec = rand.randn(TrainLen)
        H = np.sqrt(np.abs((TrainSerie - Mu)/Epsilon_vec))

        self.Mu = Mu
        self.Alpha = Alpha
        self.Sigma_Sq = Sigma_Sq
        self.H = H
        print('{0}\n[INFO] Finished initialization of parameters.'.format('=' * 20 + NOW() + '=' * 20))

rwData = ReadData(SplitYear=2013)
Priors = PriorParameters(TrainSerie=rwData.train['vwretd'])