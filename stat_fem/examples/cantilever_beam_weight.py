import numpy as np
import ufl
import stat_fem
from stat_fem.covariance_functions import sqexp
import matplotlib.pyplot as plt
from firedrake import *

# Set up base FEM, which solves Poisson's equation on a square mesh
makeplot = True
nx = 101
# Scaled variable
length = 1.0
width = 0.2
mu_f = 1
rho_g = 1
delta = width/length
gamma = 0.4*delta**2
beta = 1.25
lambda_f = beta
g = gamma


mesh = RectangleMesh(nx-1, nx-1, length, width)

V = VectorFunctionSpace(mesh, "CG", 1)
rho_g_f = Constant(rho_g)
g_f = Constant(g)
f = as_vector([0, -rho_g_f*g_f])
mu = Constant(mu_f)
lambda_ = Constant(lambda_f)
Id = Identity(mesh.geometric_dimension()) # 2x2 Identity tensor
def epsilon(u):
    return 0.5*(grad(u) + grad(u).T)

def sigma(u):
    return lambda_*div(u)*Id + 2*mu*epsilon(u)    
bc = DirichletBC(V, Constant([0, 0]), 1)
u = TrialFunction(V)
v = TestFunction(V)
a = inner(sigma(u), epsilon(v))*dx
L = dot(f, v)*dx

# create rigid body modes
x, y = SpatialCoordinate(mesh)
b0 = Function(V)
b1 = Function(V)
b2 = Function(V)
b0.interpolate(Constant([1, 0]))
b1.interpolate(Constant([0, 1]))
b2.interpolate(as_vector([-y, x]))
nullmodes = VectorSpaceBasis([b0, b1, b2])
# Make sure they're orthonormal.
nullmodes.orthonormalize()
uh = Function(V)
options={"ksp_type": "cg", 
        "ksp_max_it": 100, 
        "pc_type": "gamg",
        "mat_type": "aij",
        "ksp_converged_reason": None}
solve(a == L, uh, bcs=bc, solver_parameters=options, near_nullspace=nullmodes)

displaced_coordinates = interpolate(SpatialCoordinate(mesh) + uh, V)
displacements = interpolate(uh, V)
displaced_mesh = Mesh(displaced_coordinates)

if makeplot:
    fig, axes = plt.subplots()
    # triplot(displaced_mesh, axes=axes)
    surf = tripcolor(displacements, axes=axes)
    axes.set_aspect("equal");
    fig.colorbar(surf)
    plt.show()

# Create some fake data that is systematically different from the FEM solution.
# note that all parameters are on a log scale, so we take the true values
# and take the logarithm

# forcing covariance parameters (taken to be known)
sigma_f = np.log(2.e-2)
l_f = np.log(0.354)

# model discrepancy parameters (need to be estimated)
rho = np.log(0.7)
sigma_eta = np.log(1.e-2)
l_eta = np.log(0.5)

# data statistical errors (taken to be known)
sigma_y = 2.e-3
datagrid = 6
ndata = datagrid**2

# create fake data on a grid
x_data = np.zeros((ndata, 2))
count = 0
for i in range(datagrid):
    for j in range(datagrid):
        x_data[count, 0] = float(i+1)/float(datagrid + 1) * length
        x_data[count, 1] = float(j+1)/float(datagrid + 1) * width
        count += 1

# fake data is the true FEM solution, scaled by the mismatch factor rho, with
# correlated errors added due to model/data discrepancy and uncorrelated measurement
# errors both added to the data
y = (-gamma * rho_g / (8 * (mu_f*(3*lambda_f+2*mu_f)/(lambda_f+mu_f)) * 0.01 )*(x_data[:,0])**2*(3*length**2+2*length*(length-x_data[:,0])+(length-x_data[:,0])**2) +
     np.random.multivariate_normal(mean = np.zeros(ndata), cov = sqexp(x_data, x_data, sigma_eta, l_eta)) +
     np.random.normal(scale = sigma_y, size = ndata))

# visualize the prior FEM solution and the synthetic data
if makeplot:
    fig, axes = plt.subplots()
    plt.tripcolor(mesh.coordinates.vector().dat.data[:,0], mesh.coordinates.vector().dat.data[:,1],
                    uh.vector().dat.data[:,1], axes = axes)
    plt.colorbar()
    plt.scatter(x_data[:,0], x_data[:,1], c = y, cmap="Greys_r")
    plt.colorbar()
    axes.set_aspect = "equal"
    plt.title("Prior FEM solution and data")
    plt.show()

# Begin stat-fem solution

# Compute and assemble forcing covariance matrix using known correlated errors in forcing

G = stat_fem.ForcingCovariance(V, [sigma_f, sigma_f], [l_f, l_f])
# G = stat_fem.ForcingCovariance(V.sub(1), sigma_f, l_f)
G.assemble()


# combine data into an observational data object using known locations, observations,
# and known statistical errors
y_data_obs = np.array([[0.0, y_data_i] for y_data_i in y]).reshape(-1,x_data.shape[1])
obs_data = stat_fem.ObsData(x_data, y_data_obs, sigma_y)
# obs_data.append(stat_fem.ObsData(x_data, np.zeros_like(y), np.zeros_like(sigma_y)))
# obs_data.append(stat_fem.ObsData(x_data, y, sigma_y))

# Use MLE (MAP with uninformative prior information) to estimate discrepancy parameters
# Should get a good estimate of these values for this example problem (if not, you
# were unlucky with random sampling!)
A = assemble(a, bcs = bc)
b = assemble(L)
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

plot_Cuy = [np.diag(Cuy)[i] for i in range(len(np.diag(Cuy))) if i % 2 != 0]
plot_muy2 = [np.diag(muy2)[i] for i in range(len(np.diag(muy2))) if i % 2 != 0]
plt.figure()
plt.tripcolor(mesh.coordinates.vector().dat.data[:,0], mesh.coordinates.vector().dat.data[:,1],
                muy.vector().dat.data[:,1])
plt.colorbar()
plt.scatter(x_data[:,0], x_data[:,1], c = plot_Cuy, cmap="Greys_r")
plt.colorbar()
plt.title("Posterior FEM solution and uncertainty")
plt.show()
