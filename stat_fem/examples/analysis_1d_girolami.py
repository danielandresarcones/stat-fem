import numpy as np
import stat_fem
import json
from stat_fem.covariance_functions import sqexp
import matplotlib.pyplot as plt
from firedrake import *
from sympy import factor_list
try:
    import matplotlib.pyplot as plt
    makeplots = True
except ImportError:
    makeplots = False

# Set up base FEM, which solves Poisson's equation on a square mesh

makeplots = False
nx = 33
# Scaled variable
mesh = UnitIntervalMesh(nx-1)

V = FunctionSpace(mesh, "CG", 1)

u = TrialFunction(V)
v = TestFunction(V)

f = Constant(np.pi**2/5.0)
x = SpatialCoordinate(mesh)

a = dot(grad(v), grad(u)) * dx
L = f * v * dx

bc_1 = DirichletBC(V, 0., 1)
bc_2 = DirichletBC(V, 0., 2)

A = assemble(a, bcs = [bc_1, bc_2])

b = assemble(L)

u = Function(V)

# Create some fake data that is systematically different from the FEM solution.
# note that all parameters are on a log scale, so we take the true values
# and take the logarithm

# forcing covariance parameters (taken to be known)
sigma_f_prior = 0.3
l_f = 0.25

# model discrepancy parameters (need to be estimated)
rho = np.log(1.0)
sigma_eta_prior = np.log(0.0225/2)
l_eta = np.log(0.5)

# data statistical errors (taken to be known)
sigma_y = sqrt(2.5e-5)
ndata = nx

# create fake data on a grid
x_data = np.zeros((ndata,1))
count = 0
for i in range(ndata):
    x_data[count,0] = float(i)/float(ndata-1)
    count += 1

# Define forces array
n_repetitions = 10
factor_linspace = np.linspace(1,85,15)
forces  = np.pi**2/5.0*factor_linspace
sigma_eta_list = sigma_eta_prior*factor_linspace/7
parameters_dict = {}

for factor in factor_linspace:

    i = 0
    for repetition in range(n_repetitions):
        i += 1
        force = np.pi**2/5.0*factor
        f = Constant(force)
        print(f"Running sample {i} of force {force}...")
        # fake data is the true FEM solution, scaled by the mismatch factor rho, with
        # correlated errors added due to model/data discrepancy and uncorrelated measurement
        # errors both added to the data
        z_mean = np.exp(rho)*(0.2*force*5.0/np.pi**2*np.sin(np.pi*x_data[:,0])+0.02*force*5.0/np.pi**2*np.sin(7*np.pi*x_data[:,0]))
        sigma_eta = sigma_eta_prior * factor
        z_cov = sqexp(x_data, x_data, sigma_eta, l_eta)
        z = (z_mean + np.random.multivariate_normal(mean = np.zeros(ndata), cov = z_cov))
        y = (z + np.random.normal(scale = sigma_y, size = ndata))
        # visualize the prior FEM solution and the synthetic data


        # Begin stat-fem solution

        # Compute and assemble forcing covariance matrix using known correlated errors in forcing

        sigma_f = sigma_f_prior*factor/7
        G = stat_fem.ForcingCovariance(V, [sigma_f], [l_f])
        G.assemble()

        # combine data into an observational data object using known locations, observations,
        # and known statistical errors

        obs_data = stat_fem.ObsData(x_data, y, sigma_y)

        # Use MLE (MAP with uninformative prior information) to estimate discrepancy parameters
        # Should get a good estimate of these values for this example problem (if not, you
        # were unlucky with random sampling!)

        # ls = stat_fem.estimate_params_MAP(A, b, G, obs_data)
        ls, samples = stat_fem.estimate_params_MCMC(A, b, G, obs_data, stabilise = True, progress = False, n_samples = 20000)

        print("Parameter estimates:")
        print(ls.params)
        print("Actual input parameters:")
        true_values = np.array([rho, sigma_eta, l_eta])
        print(true_values)

        if makeplots:
            figure, axes = plt.subplots(3)
            names = [r"$\rho$",r"$\sigma_d$",r"$l_d$"]
            p_names = [r"$p(\rho)$",r"$p(\sigma_d$)",r"p($l_d$)"]
            for i in range(3):
                axes[i].hist(np.exp(samples[:, i]), 100, color='k', histtype="step")
                axes[i].set(xlabel = names[i], ylabel = p_names[i])
                axes[i].axvline(x=np.exp(ls.params[i]), c='b',linestyle = '-', label = "Estimate")
                axes[i].axvline(x=np.exp(true_values[i]), c='r',linestyle = '--', label = "True")
                axes[i].legend()
            axes[2].set(xlim = (-1,1))


        # Estimation function returns a re-usable LinearSolver object, which we can use to compute the
        # posterior FEM solution conditioned on the data

        # solve for posterior FEM solution conditioned on data
        mu, Cu = ls.solve_prior()
        mu_f, Cu_f = ls.solve_prior_generating()
        muy = Function(V)

        # plot priors 
        if makeplots:
            plt.figure()
            plt.plot(x_data[:,0],mu,'o-',markersize = 2,label = "u")
            plt.fill_between(x_data[:,0],mu+1.96*np.diag(Cu), mu-1.96*np.diag(Cu), label = "u 95 confidence",alpha = 0.5)
            plt.plot(x_data,z_mean, '+-', label = "True z")
            plt.fill_between(x_data[:,0],z_mean+1.96*np.sqrt(np.diag(z_cov)),z_mean-1.96*np.sqrt(np.diag(z_cov)), label = "True z 95 confidence", alpha = 0.5)
            plt.plot(x_data,y, '+', label = "data")
            plt.xlabel("x [m]")
            plt.ylabel("y")
            plt.legend()
            plt.title("Prior solutions")

        # solve_posterior computes the full solution on the FEM grid using a Firedrake function
        # the scale_mean option will ensure that the output is scaled to match
        # the data rather than the FEM soltuion

        ls.solve_posterior(muy, scale_mean=True)

        # covariance can only be computed for a select number of locations as covariance is a dense matrix
        # function returns the mean/covariance as numpy arrays, not Firedrake functions

        muy2, Cuy = ls.solve_posterior_covariance(scale_mean = True)
        mu_z2, Cu_z2 = ls.solve_posterior_real()

        # visualize posterior FEM solution and uncertainty

        if makeplots:
            plt.figure()
            plt.plot(x_data[:,0],muy2,'o-',markersize = 0, label = "u posterior")
            plt.fill_between(x_data[:,0],muy2+1.96*np.sqrt(np.diag(Cuy)), muy2-1.96*np.sqrt(np.diag(Cuy)), label = "u 95 confidence", alpha = 0.5)
            plt.plot(x_data[:,0],mu_z2,'o-',markersize = 2,label = "y from FEM (z+noise) posterior")
            plt.fill_between(x_data[:,0],mu_z2+1.96*np.sqrt(np.diag(Cu_z2)), mu_z2-1.96*np.sqrt(np.diag(Cu_z2)), label = "y from FEM (z+noise) 95 confidence", alpha = 0.5)
            plt.plot(x_data,y, '+', label = "data")
            plt.title("Posterior solutions")
            plt.xlabel("x [m]")
            plt.ylabel("y")
            plt.legend()
            plt.show()
        
        if i == 1:
            force_parameters = {
                "muy2": [muy2],
                "Cuy": [Cuy],
                "mu_z2": [mu_z2],
                "Cu_z2":[Cu_z2],
                "rho":[ls.params[0]],
                "sigma_d":[ls.params[1]],
                "l_d":[ls.params[2]]
            }
        else:
            
            force_parameters["muy2"].append(muy2)
            force_parameters["Cuy"].append(Cuy)
            force_parameters["mu_z2"].append(mu_z2)
            force_parameters["Cu_z2"].append(Cu_z2)
            force_parameters["rho"].append(ls.params[0])
            force_parameters["sigma_d"].append(ls.params[1])
            force_parameters["l_d"].append(ls.params[2])

    parameters_dict[force] = {
        "muy2": (np.mean(force_parameters["muy2"], axis = 0), np.std(force_parameters["muy2"], axis = 0)),
        "Cuy": (np.mean(force_parameters["Cuy"], axis = 0), np.std(force_parameters["Cuy"], axis = 0)),
        "mu_z2":(np.mean(force_parameters["mu_z2"], axis = 0), np.std(force_parameters["mu_z2"], axis = 0)),
        "Cu_z2":(np.mean(force_parameters["Cu_z2"], axis = 0), np.std(force_parameters["Cu_z2"], axis = 0)),
        "rho":(np.mean(force_parameters["rho"]), np.std(force_parameters["rho"])),
        "sigma_d":(np.mean(force_parameters["sigma_d"]), np.std(force_parameters["sigma_d"])),
        "l_d":(np.mean(force_parameters["l_d"]), np.std(force_parameters["l_d"]))
    }

tmp_dict = {
    "rho": [parameters_dict[force]["rho"] for force in parameters_dict.keys()],
    "sigma_d": [parameters_dict[force]["sigma_d"] for force in parameters_dict.keys()],
    "l_d": [parameters_dict[force]["l_d"] for force in parameters_dict.keys()]
    }

with open('parameters.json', 'w') as file:
     file.write(json.dumps(tmp_dict))

plt.figure()

for key, value in tmp_dict.items():
    value = np.array(value)
    plt.plot(forces, value[:,0], label = key)
    plt.fill_between(forces,value[:,0]+1.96*value[:,1],value[:,0]-1.96*value[:,1], label = key + ' 95 confidence', alpha = 0.5)

plt.legend()
plt.show()
