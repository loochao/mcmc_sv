# Importing all dependencies
import pickle
import argparse
import numpy as np
import pandas as pd
import pymc3 as pm
from datetime import datetime

def now():  return str(datetime.now())[:-7]

class ReadData:
    def __init__(self, SplitYear=2013, yyyymmdd_colname = 'caldt'):
        file_loc = r"https://raw.githubusercontent.com/jiacheng0409/mcmc_sv/master/sp_daily.csv"
        rwData = pd.read_csv(file_loc)
        yyyymmdd = rwData[yyyymmdd_colname]
        df_norm = (rwData - rwData.mean()) /rwData.std()
        df_norm[yyyymmdd_colname] = yyyymmdd

        # the following three lines make sure we use lag_1 period t-bill yields as exogenous variable
        temp = df_norm['tbill'].shift(periods=1)
        temp.iloc[0] = df_norm['tbill'].iloc[0].copy()
        df_norm['tbill_lag'] = temp

        temp = df_norm['vwretd'].shift(periods=1)
        temp.iloc[0] = df_norm['vwretd'].iloc[0].copy()
        df_norm['vwretd_lag'] = temp

        train_IDX = df_norm[yyyymmdd_colname] > SplitYear * (10 ** 4)
        self.train = df_norm[train_IDX]
        self.test = df_norm[~train_IDX]
        print('[INFO] Finished data importing.')

def exponential_model(training_data_df):
    returns = training_data_df['vwretd'].as_matrix()

    with pm.Model() as model_obj:
        nu = pm.Exponential('nu', 1./10, testval=5.)
        sigma = pm.Exponential('sigma', 1./.02, testval=.1)
        s = pm.GaussianRandomWalk('s', sigma**-2, shape=len(returns))
        volatility_process = pm.Deterministic('volatility_process', pm.math.exp(-2*s))
        r = pm.StudentT('r', nu, lam=1/volatility_process, observed=returns)
    return model_obj

def hidden_vol_model(training_data_df):
    returns = training_data_df['vwretd'].as_matrix()
    returns_lag = training_data_df['vwretd_lag'].as_matrix()
    tbill_lag = training_data_df['tbill_lag'].as_matrix()

    with pm.Model() as model_obj:
        n_obs = returns.shape[0]

        epsilon = pm.Normal(name='epsilon', sd=1.0, mu=.0, shape=n_obs)
        beta_vec = pm.Normal(name='beta_vec', sd=1.0, mu=.0, shape=3)

        alpha_0 = pm.HalfNormal(name='alpha_0',sd=1.0, shape=1)
        ar_1_para = pm.Bound(pm.Normal, lower=-1.0 + 1e-6, upper=1.0 - 1e-6)
        alpha_1 = ar_1_para('alpha_1', mu=0, sd=1)

        # sigma_nu = 1.0
        # sigma_inv = pm.ChiSquared(name='sigma_inv', nu=sigma_nu)
        # sigma_sq = pm.Deterministic(name='sigma', var=sigma_nu/sigma_inv)
        sigma_sq = pm.InverseGamma(name='sigma', alpha=2.0, shape=1)
        # v = pm.Normal(name='v',sd=pm.math.sqrt(sigma_sq),shape=n_obs)

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
                               mu=beta_vec[0] + tbill_lag[t]*beta_vec[1]+ returns_lag[t]*beta_vec[2],
                               sd=pm.math.sqrt(this_h),
                               observed=returns[t])
            r_list.append(this_r)
    print('[INFO] hidden volatility model built.')
    return model_obj

def main(split_year, n_draw, model):
    data = ReadData(split_year)
    training_data_df = data.train

    if model == 'exponential':
        model_obj = exponential_model(training_data_df)
    elif model == 'hidden_vol':
        model_obj= hidden_vol_model(training_data_df)
    else:
        raise NotImplementedError

    with model_obj: trace = pm.sample(draws=n_draw)
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
    print('[INFO] Code starts running at {}.\n'.format(now()))
    split_year = 2013
    n_draw = 2000

    # Already implemented models:
    # 'exponential' : from 'http://docs.pymc.io/notebooks/stochastic_volatility.html?highlight=stochastic%20volatility'
    # 'hidden_vol': from Section 10.7 [Tsay; 2002]
    chosen_model = parser.parse_args().model
    main(split_year, n_draw, chosen_model)