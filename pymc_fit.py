# Importing all dependencies
import pickle
import argparse
import multiprocessing
import numpy as np
import pandas as pd
import pymc3 as pm
from datetime import datetime

def now():  return str(datetime.now())[:-7]

class ReadData:
    def __init__(self, StartYear = 1962, EndYear=1999):
        file_loc = r"https://raw.githubusercontent.com/jiacheng0409/mcmc_sv/master/sp500_196201_201612.csv"
        rwData = pd.read_csv(file_loc)
        yyyymmdd = rwData['DATE']
        df_norm = (rwData - rwData.mean()) /rwData.std()
        df_norm['DATE'] = yyyymmdd

        temp = df_norm['logret'].shift(periods=1)
        temp.iloc[0] = 0
        df_norm['logret_lag'] = temp

        train_IDX = (df_norm['DATE'] <= (EndYear+1) * (10 ** 4)) & (df_norm['DATE'] >= StartYear * (10 ** 4))
        self.train = df_norm[train_IDX]
        self.test = df_norm[~train_IDX]
        print('[INFO {}] finished data importing.'.format(now()))

def exponential_model(training_data_df):
    logreturns = training_data_df['logret'].as_matrix()
    with pm.Model() as model_obj:
        nu = pm.Exponential('nu', 1./10, testval=5.)
        sigma = pm.Exponential('sigma', 1./.02, testval=.1)
        s = pm.GaussianRandomWalk('s', sigma**-2, shape=len(logreturns))
        volatility_process = pm.Deterministic('volatility_process', pm.math.exp(-2*s))
        r = pm.StudentT('r', nu, lam=1/volatility_process, observed=logreturns)
    return model_obj

def hidden_vol_model(training_data_df):
    logreturns = training_data_df['logret'].as_matrix()
    logreturns_lag = training_data_df['logret_lag'].as_matrix()

    with pm.Model() as model_obj:
        n_obs = logreturns.shape[0]

        epsilon = pm.Normal(name='epsilon', sd=1.0, mu=.0, shape=n_obs)
        beta_vec = pm.Normal(name='beta_vec', sd=1.0, mu=.0, shape=2)

        alpha_0 = pm.HalfNormal(name='alpha_0',sd=1.0, shape=1)
        ar_1_para = pm.Bound(pm.Normal, lower=-1.0 + 1e-6, upper=1.0 - 1e-6)
        alpha_1 = ar_1_para('alpha_1', mu=0, sd=1)

        sigma_sq = pm.InverseGamma(name='sigma', alpha=2.0, shape=1)

        ln_h_list = list()
        r_list = list()
        for t in range(n_obs):
            if t == 0:
                this_h = pm.HalfNormal(name='h_{}'.format(t), sd=1.0,shape=1)
                this_ln_h = pm.Deterministic(name='ln_h_{}'.format(t),
                                             var=pm.math.log(this_h))
            else:
                this_ln_h = pm.Normal(name='ln_h_{}'.format(t),
                                      mu=alpha_0 + alpha_1*this_ln_h,
                                      sd=pm.math.sqrt(sigma_sq),
                                      shape=1)
                this_h = pm.Deterministic(name='h_{}'.format(t),
                                          var=pm.math.exp(this_ln_h))

            ln_h_list.append(this_ln_h)
            this_r = pm.Normal(name='r_{}'.format(t),
                               mu=beta_vec[0] + logreturns_lag[t]*beta_vec[1],
                               sd=pm.math.sqrt(this_h),
                               observed=logreturns[t])
            r_list.append(this_r)
    print('[INFO {}] hidden volatility model built.'.format(now()))
    return model_obj

def main(StartYear, EndYear, n_draw, model):
    data = ReadData(StartYear, EndYear)
    training_data_df = data.train

    if model == 'exponential':
        model_obj = exponential_model(training_data_df)
    elif model == 'hidden_vol':
        model_obj= hidden_vol_model(training_data_df)
    else:
        raise NotImplementedError

    n_cpus = multiprocessing.cpu_count()
    print('[INFO {}] starts sampling on {} CPUs.'.format(now(), n_cpus))
    with model_obj: trace = pm.sample(draws=1, njobs=n_cpus)
    pm.summary(trace)

    output_file = '{}_model_trace.pkl'.format(model)
    with open(output_file,'wb') as output_file_obj:
        pickle.dump(trace,output_file_obj)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Conducting MCMC.')

    parser.add_argument('-m', action='store', dest='model', default='hidden_vol',
                        help='This argument helps specifies which model you wish to run MCMC on.\n')

    # Initialize random number generator
    np.random.seed(123)
    print('[INFO {}] code starts running.\n'.format(now()))
    StartYear = 1962
    EndYear = 1999
    n_draw = 200

    # Already implemented models:
    # 'exponential' : from 'http://docs.pymc.io/notebooks/stochastic_volatility.html?highlight=stochastic%20volatility'
    # 'hidden_vol': from Section 10.7 [Tsay; 2002]
    chosen_model = parser.parse_args().model
    main(StartYear, EndYear, n_draw, chosen_model)