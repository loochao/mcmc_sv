# -*- coding: utf-8 -*-
"""
Created on Fri May 26 02:55:27 2017
Last edited on Fri Sep 22 15:08:44 2017

@author: Jiacheng Z
"""

from __future__ import print_function
import re
import argparse
import numpy as np
import numpy.random as rand
import pandas as pd
from datetime import datetime
from numpy.linalg import inv as invert
from scipy.stats import gamma
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_spd_matrix
from arch import arch_model

def NOW(): return str(datetime.now())[:-7]
def NOWDIGIT(): return re.sub(pattern='[-: ]*', repl="", string=NOW())

def Standardize(dfSerie):
    STD = dfSerie.std()
    MEAN = dfSerie.mean()
    return (dfSerie - MEAN) / STD

class ReadData:
    def __init__(self, SplitYear=2013):
        file_loc = r"https://raw.githubusercontent.com/jiacheng0409/mcmc_sv/master/sp_daily.csv"
        raw_df = pd.read_csv(file_loc)
        yyyymmdd = raw_df['caldt']
        normalized_df = (raw_df-raw_df.mean()) / raw_df.std()
        normalized_df['caldt'] = yyyymmdd
        normalized_df['tbill_lag'] = normalized_df['tbill'].shift(1)
        normalized_df.loc[0,'tbill_lag'] =normalized_df.loc[0,'tbill']

        normalized_df['vwretd_lag'] = normalized_df['vwretd'].shift(1)
        normalized_df.loc[0, 'vwretd_lag'] = normalized_df.loc[0, 'vwretd']
        normalized_df['constant'] = np.ones(normalized_df.shape[0])

        train_index = normalized_df['caldt'] > SplitYear * (1000)
        self.train = normalized_df[train_index]
        self.test = normalized_df[~train_index]
        print('{0}\n[INFO] Finished data importing.'.format('=' * 20 + NOW() + '=' * 20))

class PriorParameters:
    def __init__(self, response_name, covariates_names, train_df, Seed=rand.randint(1)):
        self.response_name = response_name
        self.covariates_names = covariates_names
        rand.seed(Seed)
        n_obs = train_df.shape[0]
        def beta_prior():
            dimension = len(self.covariates_names)
            response = train_df.as_matrix([self.response_name]).reshape((n_obs,1))
            covariates = train_df.as_matrix([self.covariates_names]).reshape((n_obs,dimension))

            linear_model = LinearRegression(fit_intercept=False)
            fitted = linear_model.fit(X=covariates, y=response)
            mean_vec = fitted.coef_[0]

            cov_mat = make_spd_matrix(n_dim=dimension) # the covariance matrix must be SPD

            Beta = dict()
            Beta['Value'] = rand.multivariate_normal(mean=mean_vec, cov=cov_mat)
            Beta['Mean'] = mean_vec
            Beta['Cov'] = cov_mat
            return Beta

        self.Beta = beta_prior()

        garch = arch_model(train_df['vwretd'], p=1, q=1)
        fitted = garch.fit(update_freq=0,show_warning=True)
        print('[INFO] Finished fitting a GARCH model as initialization for latent volatilities:\n{}'.format(fitted.summary()))

        alpha_mean_vec = np.array([fitted.params['omega'],fitted.params['beta[1]']])
        h_mean_vec = fitted.conditional_volatility
        self.H =  h_mean_vec

        def AlphaPrior():
            mean_vec = alpha_mean_vec
            cov_mat = make_spd_matrix(n_dim=2)  # the covariance matrix must be SPD
            Alpha = dict()
            Alpha['Value'] = rand.multivariate_normal(mean=mean_vec, cov=cov_mat)
            Alpha['Mean'] = mean_vec
            Alpha['Cov'] = cov_mat
            return Alpha
        Alpha = AlphaPrior()
        # this abs(Alpha_2) <= 1 constraint makes sure that our AR(1) for volatility is stationary
        while np.abs(Alpha['Value'][1] >= 1):  Alpha = AlphaPrior()
        self.Alpha = Alpha

        def SigmaPrior():
            Lambda = 0.2
            m = 5
            DegreeOfFreedom = n_obs + m - 1
            sigma_sq_inv = rand.chisquare(DegreeOfFreedom)
            sigma_sq = dict()
            sigma_sq['Value'] = float(m * Lambda) / sigma_sq_inv
            sigma_sq['Lambda'] = Lambda
            sigma_sq['m'] = m
            return sigma_sq
        Sigma_Sq = SigmaPrior()
        self.Sigma_Sq = Sigma_Sq
        print('{0}\n[INFO] Finished initialization of parameters.'.format('=' * 20 + NOW() + '=' * 20))


def UpdateParameters(Parameters, response, covariates):
    # build normalized matrices for Eq. (10.22) in Page 419 in [Tsay; 2002]
    H_vec = Parameters.H.reshape((response.shape[0],1))
    covariates_O = covariates/np.sqrt(H_vec)
    r_O = response/np.sqrt(H_vec)

    # this following updating algorithm comes from Page 419 in [Tsay; 2002]
    def UpdateBeta():
        OldMean = Parameters.Beta['Mean']
        OldMean = OldMean.reshape((OldMean.shape[0],1))
        OldCov = Parameters.Beta['Cov']

        NewCov = invert(covariates_O.T.dot(covariates_O)+invert(OldCov))
        NewMean = NewCov.dot(covariates_O.T.dot(r_O) + invert(OldCov).dot(OldMean))
        NewMean = NewMean.reshape((NewMean.shape[0]))

        NewValue = rand.multivariate_normal(mean=NewMean, cov=NewCov)

        NewBeta = {'Value': NewValue, 'Mean': NewMean, 'Cov': NewCov}
        return NewBeta
    Parameters.Beta = UpdateBeta()

    # build constant-augmented matrices for alpha's posterior in Page 420 in [Tsay; 2002]
    Log_H = np.log(Parameters.H)
    Log_Lag_H = np.log(Parameters.H.shift(1))
    Log_Lag_H[0] = Log_H[0]

    Z_O = np.column_stack((np.ones_like(Log_Lag_H), Log_Lag_H))
    Log_H = Log_H.reshape((Log_H.shape[0],1))
    Log_Lag_H = Log_Lag_H.reshape((Log_Lag_H.shape[0],1))

    # this following updating algorithm comes from Page 420 in [Tsay; 2002]
    def UpdateAlpha():
        OldMean = Parameters.Alpha['Mean']
        OldMean = OldMean.reshape((OldMean.shape[0], 1))
        OldCov = Parameters.Alpha['Cov']
        Sigma_Sq = Parameters.Sigma_Sq['Value']+.0

        NewCov = invert(Z_O.T.dot(Z_O)/Sigma_Sq + invert(OldCov))
        NewMean = NewCov.dot(Z_O.T.dot(Log_H)/Sigma_Sq + invert(OldCov).dot(OldMean))
        NewMean = NewMean.reshape((NewMean.shape[0]))

        NewValue = rand.multivariate_normal(mean=NewMean, cov=NewCov)
        NewAlpha = {'Value': NewValue, 'Mean': NewMean, 'Cov': NewCov}
        return NewAlpha
    Parameters.Alpha = UpdateAlpha()

    # this following updating algorithm comes from Page 420 in [Tsay; 2002]
    def UpdateSigma():
        Alpha = Parameters.Alpha['Value']
        Lambda = Parameters.Sigma_Sq['Lambda']
        m = Parameters.Sigma_Sq['m']
        v = Log_H - Alpha[0] - Alpha[1] * Log_Lag_H
        Numerator = m * Lambda + np.sum(np.square(v))
        Chi2Draw = rand.chisquare(df=m + len(v) - 1)
        NewValue = Numerator / Chi2Draw
        NewSigma_Sq = Parameters.Sigma_Sq.copy()
        NewSigma_Sq['Value'] = NewValue
        return NewSigma_Sq
    Parameters.Sigma_Sq = UpdateSigma()

    # this following updating algorithm comes from Eq. (10.23) of Page 420 in [Tsay; 2002]
    def UpdateH():
        Alpha = Parameters.Alpha['Value']
        HVec = Parameters.H[:]

        # calculated the PI value for Metropolis-Hastings
        def CalcPI(H_This, H_Minus, H_Plus):
            PART1 = (response[idx] - covariates[idx].dot(Parameters.Beta['Value'])) ** 2 / (2 * H_This)
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
            # see updating scheme on Page 419 of Tsay
            Pi_Old = CalcPI(H_This, H_Minus, H_Plus)
            Q_Old = gamma.pdf(H_This, 1)

            H_Draw = rand.gamma(1)
            Pi_New = CalcPI(H_Draw, H_Minus, H_Plus)
            Q_New = gamma.pdf(H_Draw, 1)

            if Q_New * Pi_Old <=1e-10: # for numerical stability
                AcceptProbability = 1
            else:
                AcceptProbability = min([Pi_New * Q_Old / (Pi_Old * Q_New), 1])
            if rand.uniform(low=0, high=1) <= AcceptProbability: HVec[idx] = H_Draw
        return HVec
    Parameters.H = UpdateH()
    return Parameters

def main(response_name, covariates_names, NRound, NTrial):
    # -------------Data Preparations-----------
    raw_df = ReadData(SplitYear=2013)
    train_df = raw_df.train
    test_df = raw_df.test

    # -------------Initializing Priors----------
    n_obs = train_df.shape[0]
    Priors = PriorParameters(response_name, covariates_names, train_df, Seed=1)

    # TODO: Explore possibilities of parallel running
    # -----------------Training----------------
    AverageContainer = {'Alpha': np.empty(shape=(NTrial, Priors.Alpha['Value'].shape[0])),
                        'Beta': np.empty(shape=(NTrial, Priors.Beta['Value'].shape[0])),
                        'Sigma_Sq': np.empty(shape=NTrial),
                        'H': np.empty(shape=NTrial)}

    MSE = np.sum(np.square(train_df[response_name] - np.mean(train_df[response_name])))/n_obs

    response = train_df[response_name].as_matrix().reshape((n_obs,1))
    covariates = train_df[covariates_names].as_matrix()

    for trial in range(NTrial):

        RoundCount = 0
        MSE_Update = 1.0

        while RoundCount <= NRound or MSE_Update<1e-6:
            Priors = UpdateParameters(Priors, response, covariates)
            # -------------Calculate the current RE, Sum_RE, MMSE and R_Sq
            beta = Priors.Beta['Value']
            beta = beta.reshape((beta.shape[0],1))
            Fitted_Vec = covariates.dot(beta).reshape((n_obs)) + np.sqrt(Priors.H) * rand.randn(n_obs)

            Old_MSE = MSE
            RESID = response.reshape((n_obs)) - Fitted_Vec
            Sq_RESID = np.square(RESID)
            MSE = np.mean(Sq_RESID)
            MSE_Update = np.abs(MSE - Old_MSE)

            R_Sq = metrics.r2_score(y_true=response.reshape((n_obs)), y_pred=Fitted_Vec)
            RoundCount += 1

            print('{0}\n[INFO] Finished {1}th round of updating parameters using MCMC with:\n * Mean Squared Error={2};\n * R2={3}%;\n'.
                format('=' * 20 + NOW() + '=' * 20, RoundCount, MSE, 100 * R_Sq))

        if RoundCount > NRound:
            print('{0}\n[INFO] Successfully finished {1}th trial with convergence. Final in-sample statistics are:\n* Mean Squared Error={2}\n * R2={3}%\n'.
                format('=' * 20 + NOW() + '=' * 20, trial, MSE, 100 * R_Sq))
        else:
            print('{0}\n[WARNING] Exit {1}th trial without convergence. Final in-sample statistics are:\n * Mean Squared Error={2}\n * R2={3}%\n'.
                format('=' * 20 + NOW() + '=' * 20, trial, MSE, 100 * R_Sq))

        AverageContainer['Alpha'][trial] = Priors.Alpha['Value']
        AverageContainer['Beta'][trial] = Priors.Beta['Value']
        AverageContainer['Sigma_Sq'][trial] = Priors.Sigma_Sq['Value']
        AverageContainer['H'][trial] = Priors.H[-1]

    OptimalParameters = {
        'Alpha': np.mean(AverageContainer['Alpha'], axis=0),
        'Beta': np.mean(AverageContainer['Beta'], axis=0),
        'Sigma_Sq': np.mean(AverageContainer['Sigma_Sq']),
        'H': np.mean(AverageContainer['H'])}

    print('{0}\n[INFO] Training results:{1}'.format('=' * 20 + NOW() + '=' * 20, OptimalParameters))


    #---------------Prediction----------------
    TestLen = test_df.shape[0]
    Epsilon_vec = rand.randn(TestLen)

    # this following initialization of H comes from Eq. (10.20) in [Tsay; 2002]
    Alpha_0, Alpha_1 = OptimalParameters['Alpha']
    Beta_0, Beta_1 = OptimalParameters['Beta']

    response = [1] * TestLen
    temp = test_df['vwretd'].shift(periods=1)
    temp.iloc[0] = test_df['vwretd'].iloc[0].copy()
    X_test_Vec = temp

    for idTrial in range(100):
        H_Last = OptimalParameters['H']
        for idx in range(TestLen):
            V_t = (OptimalParameters['Sigma_Sq']**0.5)*rand.randn()
            H = np.exp(Alpha_0 + np.log(H_Last) * Alpha_1 + V_t)
            A = np.sqrt(H) * Epsilon_vec[idx]
            response[idx] += Beta_0 + Beta_1 * X_test_Vec[idx] + A
            H_Last = H

    #--------Evaluating the model: Calculation of MMSE-----------------
    response = [element/100.0 for element in response]
    RESID = test_df['vwretd']-response
    Sq_RESID = np.square(RESID)
    MMSE = np.mean(Sq_RESID)
    # --------Evaluating the model: Calculation of R_Sq-----------------
    SqSum_RESID = np.sum(Sq_RESID)
    SqSum_TOTAL = np.sum(np.square( test_df['vwretd'] - np.mean( test_df['vwretd'])))
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
    assert (NRound > 0) and (NTrial > 0), '[ERROR] Please give a valid simulation iterations command (i.e. must be positive)!'
    response_name = 'vwretd'
    covariates_names = ['constant', 'vwretd_lag', 'tbill_lag']

    main(response_name, covariates_names, NRound, NTrial)
