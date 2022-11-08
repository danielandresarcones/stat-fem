import numpy as np
from ufl import TrialFunction, TestFunction
from ufl import SpatialCoordinate, dx, pi, sin, dot, grad
import dolfinx.fem
from dolfinx.fem.petsc import PETSc
from dolfinx.fem import FunctionSpace, Function, dirichletbc
from petsc4py.PETSc import ScalarType
import sys

sys.path.append("/home/darcones/firedrake/stat-fem")
import stat_fem
from stat_fem.covariance_functions import sqexp
from mpi4py import MPI
try:
    import matplotlib.pyplot as plt
    makeplots = True
except ImportError:
    makeplots = False

# Set up base FEM, which solves Poisson's equation on a square mesh
class ExperimentPoisson1D:

    def __init__(self, nx) -> None:
        
        self.nx = nx

        self.mesh = dolfinx.mesh.create_unit_interval(MPI.COMM_WORLD, nx - 1)
        self.V = FunctionSpace(self.mesh, ("CG", 1))

        dim = self.mesh.topology.dim - 1

        facets = dolfinx.mesh.locate_entities_boundary(self.mesh, dim=dim,
                                            marker=lambda x: np.logical_or(np.isclose(x[0], 0.0),
                                                                            np.isclose(x[0], 1.0)))
        dofs = dolfinx.fem.locate_dofs_topological(V=self.V, entity_dim=dim, entities=facets)
        self.bcs = dirichletbc(value=ScalarType(0), dofs=dofs, V=self.V)

class ProblemPoisson:

    def __init__(self, experiment, force) -> None:

        u = TrialFunction(experiment.V)
        v = TestFunction(experiment.V)

        f = Function(experiment.V)
        x = SpatialCoordinate(experiment.mesh)
        f = force

        self.a = dolfinx.fem.form((dot(grad(v), grad(u))) * dx)
        self.L = dolfinx.fem.form(f * v * dx)
        self.bcs = experiment.bcs
        self.A = dolfinx.fem.petsc.assemble_matrix(self.a, bcs = [experiment.bcs])
        self.A.assemble()

        self.b = dolfinx.fem.petsc.assemble_vector(self.L)
        dolfinx.fem.petsc.apply_lifting(self.b, [self.a], bcs=[[experiment.bcs]])
        self.b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        dolfinx.fem.petsc.set_bc(self.b, [experiment.bcs])

        self.u = Function(experiment.V)

        self.lproblem = dolfinx.fem.petsc.LinearProblem(self.a, self.L, u=self.u, bcs=[experiment.bcs], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
        
    def solve(self):
        self.u = self.lproblem.solve()
        return self.u

nx = 66
force = np.pi**2/5.0

experiment = ExperimentPoisson1D(nx)
problem = ProblemPoisson(experiment, force)
problem.solve()

# Create some fake data that is systematically different from the FEM solution.
# note that all parameters are on a log scale, so we take the true values
# and take the logarithm

# forcing covariance parameters (taken to be known)
sigma_f = 0.3
l_f = 0.25

# model discrepancy parameters (need to be estimated)
rho = np.log(1.0)
sigma_eta = np.log(0.0225/2)
l_eta = np.log(0.5)

# data statistical errors (taken to be known)
sigma_y = np.sqrt(2.5e-5)
ndata = nx

# statfem parameters
statfem_param = dict()
statfem_param['sigma_f'] = [sigma_f]
statfem_param['l_f'] = [l_f]
statfem_param['true_rho'] = rho
statfem_param['true_sigma_eta'] = sigma_eta
statfem_param['true_l_eta'] = l_eta
statfem_param['sigma_y'] = sigma_y
statfem_param['inference_mode'] = 'MCMC'
statfem_param['stabilise'] = True

# create fake data on a grid
x_data = np.zeros((ndata,1))
count = 0
for i in range(ndata):
    x_data[count,0] = float(i)/float(ndata-1)
    count += 1

# fake data is the true FEM solution, scaled by the mismatch factor rho, with
# correlated errors added due to model/data discrepancy and uncorrelated measurement
# errors both added to the data
z_mean = np.exp(rho)*(0.2*force*5.0/np.pi**2*np.sin(np.pi*x_data[:,0])+0.02*force*5.0/np.pi**2*np.sin(7*np.pi*x_data[:,0]))
# z_mean = np.exp(rho)*(0.2*np.sin(np.pi*x_data[:,0])+0.02*np.sin(7*np.pi*x_data[:,0]))
z_cov = sqexp(x_data, x_data, sigma_eta, l_eta)
z = (z_mean + np.random.multivariate_normal(mean = np.zeros(ndata), cov = z_cov))
y = (z + np.random.normal(scale = sigma_y, size = ndata))
# visualize the prior FEM solution and the synthetic data



statfem_problem = stat_fem.StatFEMProblem(problem, experiment, x_data, y, parameters=statfem_param)
statfem_problem.solve()

# plt.figure()
# plt.plot(x_data[:,0],statfem_problem.mu,'o-',markersize = 2,label = "u")
# plt.fill_between(x_data[:,0],statfem_problem.mu+1.96*np.diag(statfem_problem.Cu), statfem_problem.mu-1.96*np.diag(statfem_problem.Cu), label = "u 95 confidence",alpha = 0.5)
# plt.plot(x_data,z_mean, '+-', label = "True z")
# plt.fill_between(x_data[:,0],z_mean+1.96*np.sqrt(np.diag(z_cov)),z_mean-1.96*np.sqrt(np.diag(z_cov)), label = "True z 95 confidence", alpha = 0.5)
# plt.plot(x_data,y, '+', label = "data")
# plt.xlabel("x [m]")
# plt.ylabel("y")
# plt.legend()
# plt.title("Prior solutions")
# plt.show()