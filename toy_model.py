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

def Standardize(dfSerie):
    STD = dfSerie.std()
    MEAN = dfSerie.mean()
    return (dfSerie-MEAN)/STD

class ReadData:
    def __init__(self,SplitYear=2013):
        file_loc = r"https://raw.githubusercontent.com/jiacheng0409/mcmc_sv/master/sp_daily.csv"
        rwData = pd.read_csv(file_loc)
        rwData['vwretd'] = Standardize(rwData['vwretd'])
        rwData['tbill'] = Standardize(rwData['tbill'])

        train_IDX = rwData['caldt'] > SplitYear*(10**4)
        self.train = rwData[train_IDX]
        self.test = rwData[~train_IDX]
        print('{0}\n[INFO] Finished data importing.'.format('='*20+NOW()+'='*20))

class JointlyGussian:
    def __init__(self,Value,Mean,Cov):
        self.Value = Value
        self.Mean = Mean
        self.Cov = Cov

class PriorParameters:
    def __init__(self, TrainData,Seed = rand.randint(1)):
        rand.seed(Seed)
        TrainLen = TrainData.shape[0]

        def BetaPrior():
            MeanVec = rand.rand(2)
            CovMat = np.abs(rand.rand(2, 2))
            Beta = JointlyGussian(Value=rand.multivariate_normal(mean=MeanVec, cov=CovMat,size=2),
                                  Mean=MeanVec,
                                  Cov=CovMat)
            return Beta
        Beta = BetaPrior()

        def AlphaPrior():
            MeanVec = rand.rand(2)
            CovMat = np.abs(rand.rand(2, 2))
            Alpha = JointlyGussian(Value=rand.multivariate_normal(mean=MeanVec,cov=CovMat,size=2),
                                   Mean=MeanVec,
                                   Cov=CovMat)
            return Alpha
        Alpha = AlphaPrior()
        # this abs(Alpha_2) <= 1 constraint makes sure that our AR(1) for volatility is stationary
        while np.abs(Alpha[1]>=1):  Alpha = AlphaPrior()

        def SigmaPrior():
            Lambda = rand.randn()
            m = rand.randint(low=1,high=10)
            DegreeOfFreedom = TrainLen + m -1
            sigma_sq_inv = rand.chisquare(DegreeOfFreedom)
            sigma_sq = float(m * Lambda) / sigma_sq_inv

            sigma_sq.Lamba = Lambda
            sigma_sq.m = m
            return sigma_sq
        Sigma_Sq = SigmaPrior()

        Epsilon_vec = rand.randn(TrainLen)
        # this following initialization of H comes from Eq. (10.20) in [Tsay; 2002]
        H = np.square((TrainData['vwretd'] - Beta[0] - Beta[1] * TrainData['tbill']) / Epsilon_vec)

        self.Beta = Beta
        self.Alpha = Alpha
        self.Sigma_Sq = Sigma_Sq
        self.H = H
        print('{0}\n[INFO] Finished initialization of parameters.'.format('=' * 20 + NOW() + '=' * 20))

rwData = ReadData(SplitYear=2013)
TrainDF=rwData.train[['vwretd','tbill']]
Priors = PriorParameters(TrainDF)

def UpdateParameters(Parameters=Priors):
    X_Vec = TrainDF['tbill']
    R_Vec = TrainDF['vwretd']

    def UpdateBeta():
        OldMean = Parameters.Beta.MeanVec
        OldCov = Parameters.Beta.CovMat
        # this following updating scheme comes from Page 419 in [Tsay; 2002]
        NewCov = np.invert(np.dot(np.transpose(X_Vec),X_Vec)+np.invert(OldCov))
        NewMean = np.dot(NewCov, np.dot(np.transpose(X_Vec), R_Vec) + np.dot(np.invert(OldCov),OldMean))

        NewBeta = rand.multivariate_normal(mean=NewMean,cov=NewCov,size=2)
        return NewBeta
    Parameters.Beta = UpdateBeta()

    def UpdateAlpha():
        NewAlpha = 0
        return NewAlpha
    Parameters.Alpha = UpdateAlpha()

    Parameters.Sigma_Sq
    Parameters.H