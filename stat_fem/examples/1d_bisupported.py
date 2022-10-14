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
delta = width/length
gamma = 0.4*delta**2
beta = 1.25
lambda_f = beta
g = gamma
E = 200e9
I = 1/12*0.0001
M = 500e6 
# rho_g = M/(nx-1)
rho_g = 1

mesh = IntervalMesh(nx-1, 0.0, length)

V = FunctionSpace(mesh, "CG", 1)
rho_g_f = Constant(rho_g)
g_f = Constant(g)

u = TrialFunction(V)
v = TestFunction(V)

f = Function(V)
x = SpatialCoordinate(mesh)
f = Constant(-rho_g_f*g_f)
E_f = mu_f*(3*lambda_f+2*mu_f)/(lambda_f+mu_f)
# f = Constant(-M/(E*I*(nx-1)))
mu = Constant(mu_f)
lambda_ = Constant(lambda_f)
a = dot(grad(u), grad(v)) * dx
L = f * v * dx

bc_1 = DirichletBC(V, 0., 1)
bc_2 = DirichletBC(V, 0., 2)

A = assemble(a, bcs = [bc_1,bc_2])

b = assemble(L)

u = Function(V)

options={"ksp_type": "cg", 
        "ksp_max_it": 100, 
        "pc_type": "gamg",
        "mat_type": "aij",
        "ksp_converged_reason": None}

# V.dofmap().set(nullspace_basis[0], 1.0)

solve(A, u, b, solver_parameters = options)
# solve(A, u, b)


# Create some fake data that is systematically different from the FEM solution.
# note that all parameters are on a log scale, so we take the true values
# and take the logarithm

# forcing covariance parameters (taken to be known)
sigma_f = np.log(2.e-4)
l_f = np.log(0.354)

# model discrepancy parameters (need to be estimated)
rho = np.log(0.9)
sigma_eta = np.log(1.e-6)
l_eta = np.log(0.5)

# data statistical errors (taken to be known)
sigma_y = 2.e-8
datagrid_x = 10
ndata = datagrid_x

# create fake data on a grid
x_data = np.zeros((ndata,1))
count = 0
for i in range(datagrid_x):
    x_data[count,0] = float(i+1)/float(datagrid_x + 1) * length
    count += 1

# fake data is the true FEM solution, scaled by the mismatch factor rho, with
# correlated errors added due to model/data discrepancy and uncorrelated measurement
# errors both added to the data
# y = (np.exp(rho)*np.sin(2.*np.pi*x_data[:,0])*np.sin(2.*np.pi*x_data[:,1]) +
#      np.random.multivariate_normal(mean = np.zeros(ndata), cov = sqexp(x_data, x_data, sigma_eta, l_eta)) +
#      np.random.normal(scale = sigma_y, size = ndata))
y = (np.exp(rho)*(-gamma*rho_g/(24*E_f)*x_data[:,0]*(x_data[:,0]**3-2*length*x_data[:,0]**2+length**3)) +
     np.random.multivariate_normal(mean = np.zeros(ndata), cov = sqexp(x_data, x_data, sigma_eta, l_eta)) +
     np.random.normal(scale = sigma_y, size = ndata))
# visualize the prior FEM solution and the synthetic data

if makeplots:
    plt.figure()
    # plt.tripcolor(mesh.coordinates.vector().dat.data[:],
    #               u.vector().dat.data)
    plt.plot(mesh.coordinates.vector().dat.data[:],u.vector().dat.data,'o',label = "FEM")
    # plt.colorbar(label="Displacement prior",pad=0.1)
    # plt.scatter(x_data,  c = y, cmap="Greys_r")
    # plt.colorbar(label="Displacement data")
    plt.plot(x_data,y, '+-', label = "data")
    plt.legend()
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

ls = stat_fem.estimate_params_MAP(A, b, G, obs_data)

print("MLE parameter estimates:")
print(ls.params)
print("Actual input parameters:")
print(np.array([rho, sigma_eta, l_eta]))

# Estimation function returns a re-usable LinearSolver object, which we can use to compute the
# posterior FEM solution conditioned on the data

# solve for posterior FEM solution conditioned on data

muy = Function(V)

# solve_posterior computes the full solution on the FEM grid using a Firedrake function
# the scale_mean option will ensure that the output is scaled to match
# the data rather than the FEM soltuion

ls.solve_posterior(muy, scale_mean=True)

# covariance can only be computed for a select number of locations as covariance is a dense matrix
# function returns the mean/covariance as numpy arrays, not Firedrake functions

muy2, Cuy = ls.solve_posterior_covariance()

# visualize posterior FEM solution and uncertainty

if makeplots:
    plt.figure()
    plt.tripcolor(mesh.coordinates.vector().dat.data[:,0], mesh.coordinates.vector().dat.data[:,1],
                  muy.vector().dat.data)
    plt.colorbar(label="Displacement posterior",pad=0.1)
    plt.scatter(x_data, c = np.diag(Cuy), cmap="Greys_r")
    plt.colorbar(label="Covariance $C_{u|y}$ at y")
    plt.title("Posterior FEM solution and uncertainty")
    plt.show()
