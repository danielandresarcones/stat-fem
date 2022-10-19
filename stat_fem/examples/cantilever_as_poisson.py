import numpy as np
import ufl
import stat_fem
from stat_fem.covariance_functions import sqexp
import matplotlib.pyplot as plt
from firedrake import *
try:
    import matplotlib.pyplot as plt
    makeplots = True
except ImportError:
    makeplots = False

# Set up base FEM, which solves Poisson's equation on a square mesh

makeplots = True
nx = 101
# Scaled variable
length = 1.0
width = 0.2
mu_f = 1
rho_g = 10
delta = width/length
gamma = 0.4*delta**2
beta = 1.25
lambda_f = beta
g = gamma


mesh = RectangleMesh(nx-1, nx-1, length, width)

V = FunctionSpace(mesh, "CG", 1)
rho_g_f = Constant(rho_g)
g_f = Constant(g)

u = TrialFunction(V)
v = TestFunction(V)

f = Function(V)
x = SpatialCoordinate(mesh)
f = Constant(-rho_g_f*g_f)
mu = Constant(mu_f)
lambda_ = Constant(lambda_f)
a = (inner((2*mu+lambda_)*grad(u), 0.01*grad(v))) * dx
L = f * v * dx

bc = DirichletBC(V, 0., 1)

A = assemble(a, bcs = bc)

b = assemble(L)

u = Function(V)

# options={"ksp_type": "cg", 
#         "ksp_max_it": 100, 
#         "pc_type": "gamg",
#         "mat_type": "aij",
#         "ksp_converged_reason": None}

# solve(A, u, b, solver_parameters = options)
solve(A, u, b)

# Create some fake data that is systematically different from the FEM solution.
# note that all parameters are on a log scale, so we take the true values
# and take the logarithm

# forcing covariance parameters (taken to be known)
sigma_f = np.log(0.00000000001)
l_f = np.log(3.53)

# model discrepancy parameters (need to be estimated)
rho = np.log(0.7)
sigma_eta = np.log(0.1)
l_eta = np.log(0.5)

# data statistical errors (taken to be known)
sigma_y = 2.e-3
datagrid_x = 10
datagrid_y = 1
ndata = datagrid_x*datagrid_y

# create fake data on a grid
x_data = np.zeros((ndata, 2))
count = 0
for i in range(datagrid_x):
    for j in range(datagrid_y):
        x_data[count, 0] = float(i+1)/float(datagrid_x + 1) * length
        x_data[count, 1] = float(j+1)/float(datagrid_y + 1) * width
        count += 1

# fake data is the true FEM solution, scaled by the mismatch factor rho, with
# correlated errors added due to model/data discrepancy and uncorrelated measurement
# errors both added to the data
# y = (np.exp(rho)*np.sin(2.*np.pi*x_data[:,0])*np.sin(2.*np.pi*x_data[:,1]) +
#      np.random.multivariate_normal(mean = np.zeros(ndata), cov = sqexp(x_data, x_data, sigma_eta, l_eta)) +
#      np.random.normal(scale = sigma_y, size = ndata))
z_mean = -gamma * rho_g * np.exp(rho) / (8 * (mu_f*(3*lambda_f+2*mu_f)/(lambda_f+mu_f)) * 0.01 )*(x_data[:,0])**2*(3*length**2+2*length*(length-x_data[:,0])+(length-x_data[:,0])**2)
z_cov = sqexp(x_data, x_data, sigma_eta, l_eta)
y = ( z_mean +
     np.random.multivariate_normal(mean = np.zeros(ndata), cov = z_cov) +
     np.random.normal(scale = sigma_y, size = ndata))
# visualize the prior FEM solution and the synthetic data

if makeplots:
    plt.figure()
    plt.tripcolor(mesh.coordinates.vector().dat.data[:,0], mesh.coordinates.vector().dat.data[:,1],
                  u.vector().dat.data)
    plt.colorbar(label="Displacement prior",pad=0.1)
    plt.scatter(x_data[:,0], x_data[:,1], c = y, cmap="Greys_r")
    plt.colorbar(label="Displacement data")
    plt.title("Prior FEM solution and data")

# Begin stat-fem solution

# Compute and assemble forcing covariance matrix using known correlated errors in forcing

G = stat_fem.ForcingCovariance(V, [sigma_f], [l_f])
G.assemble()

# combine data into an observational data object using known locations, observations,
# and known statistical errors

obs_data = stat_fem.ObsData(x_data, y, sigma_y)

# Use MLE (MAP with uninformative prior information) to estimate discrepancy parameters
# Should get a good estimate of these values for this example problem (if not, you
# were unlucky with random sampling!)

parameter_limits = [[-1,5],[-10,2],[-10,2]]

ls, samples = stat_fem.estimate_params_MCMC(A, b, G, obs_data, stabilise = False, parameter_limits=parameter_limits)

print("MLE parameter estimates:")
print(ls.params)
print("Actual input parameters:")
true_values = [rho, sigma_eta, l_eta]
print(np.array([rho, sigma_eta, l_eta]))

if makeplots:
    figure, axes = plt.subplots(3)
    figure.suptitle("MCMC 20000 samples")
    names = [r"$\rho$",r"$\sigma_d$",r"$l_d$"]
    p_names = [r"$p(\rho)$",r"$p(\sigma_d$)",r"p($l_d$)"]
    for i in range(3):
        axes[i].hist(np.exp(samples[:, i]), 100, color='k', histtype="step")
        axes[i].set(xlabel = names[i], ylabel = p_names[i])
        axes[i].axvline(x=np.exp(ls.params[i]), c='b',linestyle = '-', label = "Estimate")
        axes[i].axvline(x=np.exp(true_values[i]), c='r',linestyle = '--', label = "True")
        axes[i].legend()

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
    plt.plot(x_data[:,0],z_mean, '+-', label = "True z")
    plt.fill_between(x_data[:,0],z_mean+1.96*np.sqrt(np.diag(z_cov)),z_mean-1.96*np.sqrt(np.diag(z_cov)), label = "True z 95 confidence", alpha = 0.5)
    plt.plot(x_data[:,0],y, '+', label = "data")
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
    plt.tripcolor(mesh.coordinates.vector().dat.data[:,0], mesh.coordinates.vector().dat.data[:,1],
                  muy.vector().dat.data)
    plt.colorbar(label="Displacement posterior",pad=0.1)
    plt.scatter(x_data[:,0], x_data[:,1], c = np.diag(Cuy), cmap="Greys_r")
    plt.colorbar(label="Covariance $C_{u|y}$ at y")
    plt.title("Posterior FEM solution and uncertainty")

if makeplots:
    plt.figure()
    plt.plot(x_data[:,0],muy2,'o-',markersize = 0, label = "u posterior scaled")
    plt.fill_between(x_data[:,0],muy2+1.96*np.sqrt(np.diag(Cuy)), muy2-1.96*np.sqrt(np.diag(Cuy)), label = "u scaled 95 confidence", alpha = 0.5)
    plt.plot(x_data[:,0],mu_z2,'o-',markersize = 2,label = "y from FEM (z+noise) posterior")
    plt.fill_between(x_data[:,0],mu_z2+1.96*np.sqrt(np.diag(Cu_z2)), mu_z2-1.96*np.sqrt(np.diag(Cu_z2)), label = "y from FEM (z+noise) 95 confidence", alpha = 0.5)
    plt.plot(x_data[:,0],y, '+', label = "data")
    plt.title("Posterior solutions")
    plt.xlabel("x [m]")
    plt.ylabel("y")
    plt.legend()
    plt.show()
