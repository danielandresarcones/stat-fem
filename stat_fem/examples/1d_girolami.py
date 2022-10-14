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
nx = 33
# Scaled variable
mesh = UnitIntervalMesh(nx-1)

V = FunctionSpace(mesh, "CG", 1)

u = TrialFunction(V)
v = TestFunction(V)

f = Constant(np.pi**2/5)
x = SpatialCoordinate(mesh)

a = dot(grad(v), grad(u)) * dx
L = f * v * dx

bc_1 = DirichletBC(V, 0., 1)
bc_2 = DirichletBC(V, 0., 2)

A = assemble(a, bcs = [bc_1, bc_2])

b = assemble(L)

u = Function(V)

# options={"ksp_type": "cg", 
#         "ksp_max_it": 100, 
#         "pc_type": "gamg",
#         "mat_type": "aij",
#         "ksp_converged_reason": None}

# V.dofmap().set(nullspace_basis[0], 1.0)

# solve(A, u, b, solver_parameters = options)
solve(A, u, b)


# Create some fake data that is systematically different from the FEM solution.
# note that all parameters are on a log scale, so we take the true values
# and take the logarithm

# forcing covariance parameters (taken to be known)
# sigma_f = np.log(0.3)
# l_f = np.log(0.25)
sigma_f = 0.3
l_f = 0.25

# model discrepancy parameters (need to be estimated)
rho = np.log(0.80)
sigma_eta = np.log(0.0225/2)
l_eta = np.log(2.0)

# data statistical errors (taken to be known)
sigma_y = 2.5e-5
datagrid_x = nx
ndata = datagrid_x

# create fake data on a grid
x_data = np.zeros((ndata-2,1))
count = 0
for i in range(ndata-2):
    x_data[count,0] = float(i+1)/float(datagrid_x-1)
    count += 1

# fake data is the true FEM solution, scaled by the mismatch factor rho, with
# correlated errors added due to model/data discrepancy and uncorrelated measurement
# errors both added to the data
# y = (np.exp(rho)*np.sin(2.*np.pi*x_data[:,0])*np.sin(2.*np.pi*x_data[:,1]) +
#      np.random.multivariate_normal(mean = np.zeros(ndata), cov = sqexp(x_data, x_data, sigma_eta, l_eta)) +
#      np.random.normal(scale = sigma_y, size = ndata))
y = (np.exp(rho)*(0.2*np.sin(np.pi*x_data[:,0])+0.02*np.sin(7*np.pi*x_data[:,0])) +
     np.random.multivariate_normal(mean = np.zeros(ndata-2), cov = sqexp(x_data, x_data, sigma_eta, l_eta)) +
     np.random.normal(scale = sigma_y, size = ndata-2))
# visualize the prior FEM solution and the synthetic data


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

ls = stat_fem.estimate_params_MAP(A, b, G, obs_data)

print("MLE parameter estimates:")
print(ls.params)
print("Actual input parameters:")
print(np.array([rho, sigma_eta, l_eta]))

# Estimation function returns a re-usable LinearSolver object, which we can use to compute the
# posterior FEM solution conditioned on the data

# solve for posterior FEM solution conditioned on data
mu, Cu = ls.solve_prior()
mu_z, Cu_z = ls.solve_prior_generating()
muy = Function(V)

# plot priors 
if makeplots:
    plt.figure()
    plt.plot(x_data[:,0],mu,'o-',markersize = 2,label = "u")
    plt.fill_between(x_data[:,0],mu+1.96*np.diag(Cu), mu-1.96*np.diag(Cu), label = "u 95 confidence",alpha = 0.5)
    plt.plot(x_data[:,0],mu_z,'o-',markersize = 2,label = "z")
    plt.fill_between(x_data[:,0],mu_z+1.96*np.diag(Cu_z), mu_z-1.96*np.diag(Cu_z), label = "z 95 confidence",alpha = 0.5)
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

muy2, Cuy = ls.solve_posterior_covariance()
mu_z2, Cu_z2 = ls.solve_posterior_generating()

# if makeplots:
#     plt.figure()
#     plt.plot(x_data[:,0],muy2,'o-',markersize = 0,label = "u")
#     plt.fill_between(x_data[:,0],muy2+1.96*np.diag(Cuy), muy2-1.96*np.diag(Cuy), alpha = 0.5)
#     plt.plot(x_data[:,0],mu_z2,'o-',markersize = 2,label = "z")
#     plt.fill_between(x_data[:,0],mu_z2+1.96*np.diag(Cu_z2), mu_z2-1.96*np.diag(Cu_z2), alpha = 0.5)
#     plt.plot(x_data,y, '+-', label = "data")
#     plt.legend()
#     plt.title("Posterior u, Posterior z and data")

# visualize posterior FEM solution and uncertainty

if makeplots:
    plt.figure()
    plt.plot(mesh.coordinates.vector().dat.data[:],muy.vector().dat.data,'o-',markersize = 0, label = "u posterior")
    plt.fill_between(x_data[:,0],muy.vector().dat.data[1:-1]+1.96*np.diag(Cuy), muy.vector().dat.data[1:-1]-1.96*np.diag(Cuy), label = "u 95 confidence", alpha = 0.5)
    plt.plot(x_data[:,0],mu_z2,'o-',markersize = 2,label = "z posterior")
    plt.fill_between(x_data[:,0],mu_z2+1.96*np.diag(Cu_z2), mu_z2-1.96*np.diag(Cu_z2), label = "z 95 confidence", alpha = 0.5)
    plt.plot(x_data,y, '+', label = "data")
    plt.title("Posterior solutions")
    plt.xlabel("x [m]")
    plt.ylabel("y")
    plt.legend()
    plt.show()
