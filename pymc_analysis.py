import matplotlib as mpl
mpl.use('Agg')
import pickle
import numpy as np
import pymc3 as pm
import matplotlib.pyplot as plt
from pymc_fit import ReadData
trained_file_path = r"/home/jasonzou/mcmc_sv/trained_trace.pkl"
with open(trained_file_path,'rb') as trained_file_obj:
    trained_trace = pickle.load(trained_file_obj)

axes = pm.traceplot(trace=trained_trace)
plt.title('Fig 1. Parameters in the model')
axes[0][0].figure.savefig('parameters_plot.png')

plt.figure()
plt.title('Log volatility')
plt.plot(trained_trace['s'].T, 'b', alpha=.03)
plt.xlabel('Time')
plt.ylabel('Log volatility')
plt.title('Fig 2. ln(volatility)')
plt.savefig('log_volatility.png')

returns = ReadData().train['vwretd'].as_matrix()
plt.figure()
plt.plot(np.abs(returns))
plt.plot(np.exp(trained_trace['s'].T), 'r', alpha=.03)
sd = np.exp(trained_trace['s'].T)
plt.xlabel('Time')
plt.ylabel('Returns')
plt.title('Fig 3. Absolute returns and std. of volatility')
plt.savefig('absr_sd.png')