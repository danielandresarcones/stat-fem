import json
import numpy as np
import matplotlib.pyplot as plt

forces  = np.pi**2/5.0*np.linspace(1,100,25)

with open('parameters_force.json', 'r') as file:
    parameters_dict = json.load(file)

plt.figure()

for key, value in parameters_dict.items():
    value = np.array(value)
    plt.plot(forces, value[:,0], label = key)
    plt.fill_between(forces,value[:,0]+1.96*value[:,1], value[:,0]-1.96*value[:,1], label = key + ' 95 confidence', alpha = 0.5)

plt.xlabel('Force')
plt.xlim(0,200)
plt.ylabel(r'log($\theta$)')
plt.title('Mean hyperparameters with 25 runs, $n_y=33$,  using MCMC with 20000 samples and constant $\sigma_\eta$')
plt.legend()
plt.show()

