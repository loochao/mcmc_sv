# -*- coding: utf-8 -*-
"""
Created on Fri May 26 02:55:27 2017

@author: Jiacheng Z
"""

from __future__ import print_function

import argparse
import re
from datetime import datetime

import numpy as np
import numpy.random as rand
import pandas as pd
import statsmodels.api as sm
from numpy.linalg import inv as invert
from scipy.stats import gamma
from sklearn import metrics


def NOW():
    return str(datetime.now())[:-7]

def NOWDIGIT():
    return re.sub(pattern='[-: ]*', repl="", string=NOW())

def Standardize(dfSerie):
    STD = dfSerie.std()
    MEAN = dfSerie.mean()
    return (dfSerie - MEAN) / STD

class ReadData:
    def __init__(self, SplitYear=2013):
        file_loc = r"https://raw.githubusercontent.com/jiacheng0409/mcmc_sv/master/sp_daily.csv"
        rwData = pd.read_csv(file_loc)
        rwData['vwretd'] = Standardize(rwData['vwretd'])
        rwData['tbill'] = Standardize(rwData['tbill'])

        # the following three lines make sure we use lag_1 period t-bill yields as exogenous variable
        temp = rwData['tbill'].shift(periods=1)
        temp.iloc[0] = rwData['tbill'].iloc[0].copy()
        rwData['tbill'] = temp

        train_IDX = rwData['caldt'] > SplitYear * (10 ** 4)
        self.train = rwData[train_IDX]
        self.test = rwData[~train_IDX]
        print('{0}\n[INFO] Finished data importing.'.format('=' * 20 + NOW() + '=' * 20))

class PriorParameters:
    def __init__(self, TrainData, Seed=rand.randint(1)):
        rand.seed(Seed)
        TrainLen = TrainData.shape[0]

        def BetaPrior():
            # TODO: use GARCH output as initialized mean vector
            Y = TrainData['vwretd']
            X = TrainData['tbill']
            X = sm.add_constant(X)
            Model = sm.OLS(Y, X)
            Results = Model.fit()

            MeanVec = Results.params
            CovMat = [[9, 0], [0, 9]]
            Beta = dict()
            Beta['Value'] = rand.multivariate_normal(mean=MeanVec, cov=CovMat)
            Beta['Mean'] = MeanVec
            Beta['Cov'] = CovMat
            return Beta

        Beta = BetaPrior()

        def AlphaPrior():
            # TODO: use GARCH output as initialized mean vector
            MeanVec = [0.4, 0.8]
            CovMat = [[0.09, 0], [0, 0.04]]
            Alpha = dict()
            Alpha['Value'] = rand.multivariate_normal(mean=MeanVec, cov=CovMat)
            Alpha['Mean'] = MeanVec
            Alpha['Cov'] = CovMat
            return Alpha

        Alpha = AlphaPrior()
        # this abs(Alpha_2) <= 1 constraint makes sure that our AR(1) for volatility is stationary
        while np.abs(Alpha['Value'][1] >= 1):  Alpha = AlphaPrior()

        def SigmaPrior():
            Lambda = 0.2
            m = 5
            DegreeOfFreedom = TrainLen + m - 1
            sigma_sq_inv = rand.chisquare(DegreeOfFreedom)
            sigma_sq = dict()
            sigma_sq['Value'] = float(m * Lambda) / sigma_sq_inv
            sigma_sq['Lambda'] = Lambda
            sigma_sq['m'] = m

            return sigma_sq

        Sigma_Sq = SigmaPrior()

        Epsilon_vec = rand.randn(TrainLen)
        # this following initialization of H comes from Eq. (10.20) in [Tsay; 2002]
        H = np.square((TrainData['vwretd'] - Beta['Value'][0] - Beta['Value'][1] * TrainData['tbill']) / Epsilon_vec)
        H[H == 0] = 1e-5  # because we wish to calculate log of H_i's, we need to avoid zeros
        H = H.tolist()

        self.Beta = Beta
        self.Alpha = Alpha
        self.Sigma_Sq = Sigma_Sq
        self.H = H
        print('{0}\n[INFO] Finished initialization of parameters.'.format('=' * 20 + NOW() + '=' * 20))


def UpdateParameters(Parameters, TrainDF):
    R_Vec = TrainDF['vwretd']
    temp = R_Vec.shift(periods=1)
    temp.iloc[0] = R_Vec.iloc[0].copy()
    X_Vec = temp
    # X_Vec = TrainDF['tbill']
    R_Vec = TrainDF['vwretd']
    X_Mat = np.array([np.ones_like(X_Vec), X_Vec])
    Log_PriorH = np.log(Parameters.H)
    Lag1_IDX = [0] + range(len(Log_PriorH) - 1)
    Log_Lag1_PrioH = Log_PriorH[Lag1_IDX]

    def UpdateBeta():
        # this following updating algorithm comes from Page 419 in [Tsay; 2002]
        OldMean = Parameters.Beta['Mean']
        OldCov = Parameters.Beta['Cov']
        NewCov = invert(np.dot(X_Mat, np.transpose(X_Mat)) + invert(OldCov))
        NewMean = np.dot(NewCov, np.dot(X_Mat, R_Vec) + np.dot(invert(OldCov), OldMean))
        NewValue = rand.multivariate_normal(mean=NewMean, cov=NewCov)

        NewBeta = {
            'Value': NewValue,
            'Mean': NewMean,
            'Cov': NewCov
        }
        return NewBeta

    Parameters.Beta = UpdateBeta()

    def UpdateAlpha():
        # this following updating algorithm comes from Page 420 in [Tsay; 2002]
        OldMean = Parameters.Alpha['Mean']
        OldCov = Parameters.Alpha['Cov']
        Sigma_Sq = Parameters.Sigma_Sq['Value']
        Z_Mat = np.array([[1] * len(Log_Lag1_PrioH), Log_Lag1_PrioH.tolist()])
        NewCov = invert(np.dot(Z_Mat, np.transpose(Z_Mat)) / Sigma_Sq + invert(OldCov))
        NewMean = np.dot(NewCov, np.dot(Z_Mat, np.transpose(Log_PriorH)) / Sigma_Sq + np.dot(invert(OldCov), OldMean))
        NewValue = rand.multivariate_normal(mean=NewMean, cov=NewCov)

        NewAlpha = {
            'Value': NewValue,
            'Mean': NewMean,
            'Cov': NewCov
        }
        return NewAlpha

    Parameters.Alpha = UpdateAlpha()

    def UpdateSigma():
        # this following updating algorithm comes from Page 420 in [Tsay; 2002]
        Alpha = Parameters.Alpha['Value']
        Lambda = Parameters.Sigma_Sq['Lambda']
        m = Parameters.Sigma_Sq['m']
        v = Log_PriorH - Alpha[0] - Alpha[1] * Log_Lag1_PrioH
        Numerator = m * Lambda + np.sum(np.square(v))
        Chi2Draw = rand.chisquare(df=m + len(Log_PriorH) - 1)
        NewValue = Numerator / Chi2Draw
        NewSigma_Sq = Parameters.Sigma_Sq.copy()
        NewSigma_Sq['Value'] = NewValue
        return NewSigma_Sq

    Parameters.Sigma_Sq = UpdateSigma()

    def UpdateH():
        Alpha = Parameters.Alpha['Value']
        HVec = Parameters.H[:]

        def CalcPI(H_This, H_Minus, H_Plus):
            PART1 = (R_Vec.iloc[idx] - X_Vec.iloc[idx] * Parameters.Beta['Value'][1]) ** 2 / (2 * H_This)

            mu = Alpha[0] * (1 - Alpha[1]) + Alpha[1] * (np.log(H_Minus) + np.log(H_Plus)) / (1 + Alpha[1] ** 2)
            sigma_sq = Parameters.Sigma_Sq['Value'] / (1 + Alpha[1] ** 2)
            PART2 = (np.log(H_This) - mu) ** 2 / (2 * sigma_sq)

            PI = H_This ** (-1.5) * np.exp(-PART1 - PART2)
            return PI

        for idx, H_This in enumerate(HVec):
            if idx == len(HVec) - 1 or idx == 0: continue  # edge case for H_0 and H_n
            H_Minus = HVec[idx - 1]
            H_Plus = HVec[idx + 1]

            # the following acception/rejection scheme is called 'Metropolis Algorithm'
            # see updating scheme on pg. 419 of Tsay
            Pi_Old = CalcPI(H_This, H_Minus, H_Plus)
            Q_Old = gamma.pdf(H_This, 1)

            H_Draw = rand.gamma(1)
            Pi_New = CalcPI(H_Draw, H_Minus, H_Plus)
            Q_New = gamma.pdf(H_Draw, 1)

            if Q_New != 0 and Pi_Old != 0:
                AcceptProbability = 1
            else:
                AcceptProbability = min([Pi_New * Q_Old / (Pi_Old * Q_New), 1])

            if rand.uniform(low=0, high=1) <= AcceptProbability:  HVec[idx] = H_Draw

        return HVec

    Parameters.H = UpdateH()


def main(NRound, NTrial):
    # -------------Data Preparations-----------
    rwData = ReadData(SplitYear=2013)
    TrainDF = rwData.train[['vwretd', 'tbill']]
    TestDF = rwData.test[['vwretd', 'tbill']]

    # -------------Initializing Priors----------
    Priors = PriorParameters(TrainDF)

    # TODO: Explore possibilities of parallel running
    # -----------------Training----------------
    AverageContainer = {'Alpha': np.empty(shape=(NTrial, 2)),
                        'Beta': np.empty(shape=(NTrial, 2)),
                        'Sigma_Sq': np.empty(shape=NTrial),
                        'H': np.empty(shape=NTrial)
                        }
    Y_Mean = np.mean(TrainDF['vwretd'])
    TrainSqSum_TOTAL = np.sum(np.square(TrainDF['vwretd'] - Y_Mean))

    R_Vec = TrainDF['vwretd']
    temp = R_Vec.shift(periods=1)
    temp.iloc[0] = R_Vec.iloc[0].copy()
    X_Vec = temp

    for trial in range(NTrial):

        RoundCount = 0;
        Sum_RESID = 1.0;
        MSE = 1000.0;
        MSE_Update = 1.0

        while RoundCount <= NRound and np.abs(Sum_RESID) > 1e-2 and MSE_Update > 1e-2:
            UpdateParameters(Priors, TrainDF)
            # -------------Calculate the current RE, Sum_RE, MMSE and R_Sq

            B_Vec = Priors.Beta['Value'][0] + Priors.Beta['Value'][1] * X_Vec
            A_Vec = np.sqrt(Priors.H) * rand.randn(len(X_Vec))
            Fitted_Vec = B_Vec + A_Vec

            RESID = TrainDF['vwretd'] - Fitted_Vec
            Sum_RESID = np.sum(RESID)
            Sq_RESID = np.square(RESID)

            Old_MSE = MSE
            MSE = np.mean(Sq_RESID)
            MSE_Update = np.abs(MSE - Old_MSE)

            R_Sq = metrics.r2_score(y_true=TrainDF['vwretd'], y_pred=Fitted_Vec)
            RoundCount += 1

            print(
                '{0}\n[INFO] Finished {1}th round of updating parameters using MCMC with:\n * Mean Squared Error={2};\n * R2={3}%;\n'.
                format('=' * 20 + NOW() + '=' * 20, RoundCount, MSE, 100 * R_Sq))

        if RoundCount > NRound:
            print(
                '{0}\n[INFO] Successfully finished {1}th trial with convergence. Final in-sample statistics are:\n* Mean Squared Error={2}\n * R2={3}%\n'.
                format('=' * 20 + NOW() + '=' * 20, trial, MSE, 100 * R_Sq))
        else:
            print(
                '{0}\n[WARNING] Exit {1}th trial without convergence. Final in-sample statistics are:\n * Mean Squared Error={2}\n * R2={3}%\n'.
                format('=' * 20 + NOW() + '=' * 20, trial, MSE, 100 * R_Sq))

        AverageContainer['Alpha'][trial] = Priors.Alpha['Value']
        AverageContainer['Beta'][trial] = Priors.Beta['Value']
        AverageContainer['Sigma_Sq'][trial] = Priors.Sigma_Sq['Value']
        AverageContainer['H'][trial] = Priors.H[-1]
    OptimalParameters = {
        'Alpha': np.mean(AverageContainer['Alpha'], axis=0),
        'Beta': np.mean(AverageContainer['Beta'], axis=0),
        'Sigma_Sq': np.mean(AverageContainer['Sigma_Sq']),
        'H': np.mean(AverageContainer['H'])
    }
    print('{0}\n[INFO] Training results:{1}'.format('=' * 20 + NOW() + '=' * 20, OptimalParameters))


    #---------------Prediction----------------
    TestLen = TestDF.shape[0]
    Epsilon_vec = rand.randn(TestLen)

    # this following initialization of H comes from Eq. (10.20) in [Tsay; 2002]
    Alpha_0, Alpha_1 = OptimalParameters['Alpha']
    Beta_0, Beta_1 = OptimalParameters['Beta']

    R_Vec = [1] * TestLen
    temp = TestDF['vwretd'].shift(periods=1)
    temp.iloc[0] = TestDF['vwretd'].iloc[0].copy()
    X_test_Vec = temp

    for idTrial in range(100):
        H_Last = OptimalParameters['H']
        for idx in range(TestLen):
            V_t = (OptimalParameters['Sigma_Sq']**0.5)*rand.randn()
            H = np.exp(Alpha_0 + np.log(H_Last) * Alpha_1 + V_t)
            A = np.sqrt(H) * Epsilon_vec[idx]
            R_Vec[idx] += Beta_0 + Beta_1 * X_test_Vec[idx] + A
            H_Last = H

    #--------Evaluating the model: Calculation of MMSE-----------------
    R_Vec = [element/100.0 for element in R_Vec]
    RESID = TestDF['vwretd']-R_Vec
    Sq_RESID = np.square(RESID)
    MMSE = np.mean(Sq_RESID)
    # --------Evaluating the model: Calculation of R_Sq-----------------
    SqSum_RESID = np.sum(Sq_RESID)
    SqSum_TOTAL = np.sum(np.square( TestDF['vwretd'] - np.mean( TestDF['vwretd'])))
    R_Sq = 1-(SqSum_RESID/SqSum_TOTAL)
    print('{0}\n[INFO] Successfully exit the program with the following prediction results:\n * MMSE={1};\n * R_Sq={2}%'.
          format('='*20+NOW()+'='*20, MMSE, R_Sq*100))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Conducting MCMC.')

    parser.add_argument('-r', action='store', dest='NRound', default='6000',
                        help='This argument helps specifies how many iterations we run within MCMC.\n' +
                             'If you have input a decimal number, the code will take the floor int.')

    parser.add_argument('-t', action='store', dest='NTrial', default='1',
                        help='This argument helps specifies how many iterations we run the entire MCMC.\n' +
                             'If you have input a decimal number, the code will take the floor int.')

    args = parser.parse_args()
    NRound = int(args.NRound)
    NTrial = int(args.NTrial)
    assert (args.NRound > 0) and (
    args.NTrial > 0), '[ERROR] Please give a valid simulation iterations command (i.e. must be positive)!'
    main(NRound, NTrial)
