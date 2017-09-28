import matplotlib as mpl
# 'Agg' to forbid matplotlib from attempting at displaying image, which is not what we want for running this code on servers
mpl.use('Agg')

import pickle
import glob
import os
import numpy as np
import pymc3 as pm
import matplotlib.pyplot as plt
from pymc_fit import ReadData

curr_dir = os.path.dirname(os.path.realpath(__file__))
trained_file_paths = [os.path.abspath(fp) for fp in glob.glob(os.path.join(curr_dir,'*.pkl'))]
result_dir = os.path.dirname(curr_dir,'result')
if not os.path.isdir(result_dir): os.mkdir(result_dir)

for trained_file_path in trained_file_paths:
    model_name = os.path.basename(trained_file_path)[:-4]
    with open(trained_file_path,'rb') as trained_file_obj:
        trained_trace = pickle.load(trained_file_obj)

    axes = pm.traceplot(trace=trained_trace)
    axes[0][0].figure.savefig(os.path.join(result_dir,'parameters_plot_{}.png'.format(model_name)))

    plt.figure()
    plt.title('Log volatility')
    plt.plot(trained_trace['s'].T, 'b', alpha=.03)
    plt.xlabel('Time')
    plt.ylabel('Log volatility')
    plt.title('Fig 2. ln(volatility)')
    plt.savefig(os.path.join(result_dir,'log_volatility_{}.png'.format(model_name)))

    returns = ReadData().train['vwretd'].as_matrix()
    plt.figure()
    plt.plot(np.abs(returns))
    plt.plot(np.exp(trained_trace['s'].T), 'r', alpha=.03)
    sd = np.exp(trained_trace['s'].T)
    plt.xlabel('Time')
    plt.ylabel('Returns')
    plt.title('Fig 3. Absolute returns and std. of volatility')
    plt.savefig(os.path.join(result_dir,'absr_sd_{}.png'.format(model_name)))