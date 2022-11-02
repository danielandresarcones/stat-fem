import numpy as np
import ufl
import sys
sys.path.append("/home/darcones/firedrake/stat-fem")
import dolfinx
from dolfinx.fem import Function, FunctionSpace, VectorFunctionSpace, Constant, dirichletbc
import ufl
from ufl import TrialFunction, TestFunction
from ufl import SpatialCoordinate, dx, pi, sin, dot, grad, inner
from petsc4py.PETSc import ScalarType
from dolfinx.fem.petsc import PETSc
from dolfinx import fem, io
from mpi4py import MPI

import stat_fem
from stat_fem.covariance_functions import sqexp
import matplotlib.pyplot as plt
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
rho_g = 1
delta = width/length
gamma = 0.4*delta**2
beta = 1.25
lambda_f = beta
g = gamma

class ExperimentCantileverPoisson:

    def __init__(self) -> None:
        
        self.mesh = dolfinx.mesh.create_rectangle(comm = MPI.COMM_WORLD, points = [np.array([0.0,0.0]), np.array([length, width])], n=[nx-1, nx-1])

        self.V = VectorFunctionSpace(self.mesh, ("CG", 1))

        # Define boundary conditions
        def clamped_boundary(x):
            return np.isclose(x[0], 0)

        self.fdim = self.mesh .topology.dim - 1
        boundary_facets = dolfinx.mesh.locate_entities_boundary(self.mesh , self.fdim, clamped_boundary)

        u_D = np.array([0,0], dtype=ScalarType)
        self.bcs = fem.dirichletbc(u_D, fem.locate_dofs_topological(self.V, self.fdim, boundary_facets), self.V)


class ProblemCantileverPoisson:

    def __init__(self, experiment) -> None:

        ## Free of traction
        T = fem.Constant(experiment.mesh , ScalarType((0, 0)))

        ## Integration measure
        ds = ufl.Measure("ds", domain =experiment.mesh )


        # Variational formulation

        def epsilon(u):
            return ufl.sym(ufl.grad(u)) # Equivalent to 0.5*(ufl.nabla_grad(u) + ufl.nabla_grad(u).T)
        def sigma(u):
            return lambda_f * ufl.nabla_div(u) * ufl.Identity(u.geometric_dimension()) + 2*mu_f*epsilon(u)

        u = ufl.TrialFunction(experiment.V)
        v = ufl.TestFunction(experiment.V)
        f = fem.Constant(experiment.mesh , ScalarType((0, -rho_g*g)))

        self.a = ufl.inner(sigma(u), epsilon(v)) * ufl.dx
        self.L = ufl.dot(f, v) * ufl.dx + ufl.dot(T, v) * ds

        self.A = dolfinx.fem.petsc.assemble_matrix(fem.form(self.a), bcs = [experiment.bcs])
        self.A.assemble()

        self.b = dolfinx.fem.petsc.assemble_vector(fem.form(self.L))
        dolfinx.fem.petsc.apply_lifting(self.b, [fem.form(self.a)], bcs=[[experiment.bcs]])
        self.b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        dolfinx.fem.petsc.set_bc(self.b, [experiment.bcs])

        # Solve the problem
        self.lproblem = fem.petsc.LinearProblem(self.a, self.L, bcs=[experiment.bcs], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})

    def solve(self):

        # options={"ksp_type": "cg", 
        #         "ksp_max_it": 100, 
        #         "pc_type": "gamg",
        #         "mat_type": "aij",
        #         "ksp_converged_reason": None}

        self.u = self.lproblem.solve()
        return self.u

experiment = ExperimentCantileverPoisson()
problem = ProblemCantileverPoisson(experiment)
problem.solve()

# Create some fake data that is systematically different from the FEM solution.
# note that all parameters are on a log scale, so we take the true values
# and take the logarithm

# forcing covariance parameters (taken to be known)
sigma_f = np.log(0.0005)
l_f = np.log(0.353)

# model discrepancy parameters (need to be estimated)
rho = np.log(0.7)
sigma_eta = np.log(0.005)
l_eta = np.log(0.05)

# data statistical errors (taken to be known)
sigma_y = 2.e-5
datagrid_x = 25
datagrid_y = 1
ndata = datagrid_x*datagrid_y

# statfem parameters
statfem_param = dict()
statfem_param['sigma_f'] = [0.0, sigma_f]
statfem_param['l_f'] = [0.0, l_f]
statfem_param['true_rho'] = rho
statfem_param['true_sigma_eta'] = sigma_eta
statfem_param['true_l_eta'] = l_eta
statfem_param['sigma_y'] = sigma_y
statfem_param['inference_mode'] = 'MCMC'
statfem_param['stabilise'] = True

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
y = np.array( z_mean +
     np.random.multivariate_normal(mean = np.zeros(ndata), cov = z_cov) +
     np.random.normal(scale = sigma_y, size = ndata))
y = np.pad(np.expand_dims(y, axis=1), ((0,0),(1,0)), 'constant', constant_values=0.0)
# visualize the prior FEM solution and the synthetic data

with io.XDMFFile(experiment.mesh.comm, "deformation.xdmf", "w") as xdmf:
    xdmf.write_mesh(experiment.mesh)
    problem.u.name = "Deformation prior"
    xdmf.write_function(problem.u)
    
    statfem_problem = stat_fem.StatFEMProblem(problem, experiment, x_data, y, parameters=statfem_param, makeplots = False)
    statfem_problem.solve()

    statfem_problem.muy.name = "Deformation posterior"
    xdmf.write_function(statfem_problem.muy)

# plot priors 
if makeplots:
    plt.figure()
    plt.plot(x_data[:,0],statfem_problem.mu[:,1],'o-',markersize = 2,label = "u")
    plt.fill_between(x_data[:,0],statfem_problem.mu[:,1]+1.96*np.diag(statfem_problem.Cu)[1::2], statfem_problem.mu[:,1]-1.96*np.diag(statfem_problem.Cu)[1::2], label = "u 95 confidence",alpha = 0.5)
    plt.plot(x_data[:,0],z_mean, '+-', label = "True z")
    plt.fill_between(x_data[:,0],z_mean+1.96*np.sqrt(np.diag(z_cov)),z_mean-1.96*np.sqrt(np.diag(z_cov)), label = "True z 95 confidence", alpha = 0.5)
    plt.plot(x_data[:,0],y[:,1], '+', label = "data")
    plt.xlabel("x [m]")
    plt.ylabel("y")
    plt.legend()
    plt.title("Prior solutions")

if makeplots:
    plt.figure()
    plt.tripcolor(experiment.mesh.geometry.x[:,0], experiment.mesh.geometry.x[:,1],
                  np.reshape(statfem_problem.muy.vector, (-1,2))[:,1])
    plt.colorbar(label="Displacement posterior",pad=0.1)
    plt.scatter(x_data[:,0], x_data[:,1], c = np.diag(statfem_problem.Cuy)[1::2], cmap="Greys_r")
    plt.colorbar(label="Covariance $C_{u|y}$ at y")
    plt.title("Posterior FEM solution and uncertainty")

if makeplots:
    plt.figure()
    plt.plot(x_data[:,0],statfem_problem.muy2[:,1],'o-',markersize = 0, label = "u posterior scaled")
    plt.fill_between(x_data[:,0],statfem_problem.muy2[:,1]+1.96*np.sqrt(np.diag(statfem_problem.Cuy)[1::2]), statfem_problem.muy2[:,1]-1.96*np.sqrt(np.diag(statfem_problem.Cuy)[1::2]), label = "u scaled 95 confidence", alpha = 0.5)
    plt.plot(x_data[:,0],statfem_problem.mu_z2[:,1],'o-',markersize = 2,label = "y from FEM (z+noise) posterior")
    plt.fill_between(x_data[:,0],statfem_problem.mu_z2[:,1]+1.96*np.sqrt(np.diag(statfem_problem.Cu_z2)[1::2]), statfem_problem.mu_z2[:,1]-1.96*np.sqrt(np.diag(statfem_problem.Cu_z2)[1::2]), label = "y from FEM (z+noise) 95 confidence", alpha = 0.5)
    plt.plot(x_data[:,0],y[:,1], '+', label = "data")
    plt.title("Posterior solutions")
    plt.xlabel("x [m]")
    plt.ylabel("y")
    plt.legend()
    plt.show()
