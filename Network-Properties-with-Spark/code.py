import numpy as np
import matplotlib.pyplot as plt 
import scipy.stats as st

samples = 20

X = np.linspace(0, 10, samples)

u1 = 4
s1 = 0.8
u_prior = st.norm(u1, s1).pdf(X)


ux = 6
sx = 1.5
u_sample = dist = st.norm(ux, sx).pdf(X)


xt = st.norm(ux, sx).rvs(samples)
var_n = 1 / ((1/s1**2) + (samples/sx**2))
un = var_n * ((u1/s1**2) + (np.mean(xt) * samples/sx**2))
print('Mean of posterior distribution is '+str(un)+' and variance is '+str(var_n))


u_posterior = st.norm(un, np.sqrt(var_n)).pdf(X)


plt.plot(X, u_prior, 'b-', label='prior')
plt.plot(X, u_sample, 'g-', label='sample')
plt.plot(X, u_posterior, 'r-', label='posterior')
plt.legend(loc='upper left')
plt.title('Probability Density Plot')
plt.xlabel('X')
plt.ylabel('Probability Density')
plt.show()