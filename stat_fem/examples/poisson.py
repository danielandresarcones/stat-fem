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

class ExperimentPoisson:

    def __init__(self) -> None:
        
        nx = 101

        self.mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, nx - 1, nx - 1)
        self.V = FunctionSpace(self.mesh, ("CG", 1))

        

        facets = dolfinx.mesh.locate_entities_boundary(self.mesh, dim=1,
                                            marker=lambda x: np.logical_or.reduce((np.isclose(x[0], 0.0),
                                                                                    np.isclose(x[0], 1.0),
                                                                                    np.isclose(x[1], 0.0),
                                                                                    np.isclose(x[1], 1.0))))
        dofs = dolfinx.fem.locate_dofs_topological(V=self.V, entity_dim=1, entities=facets)
        self.bcs = dirichletbc(value=ScalarType(0), dofs=dofs, V=self.V)

class ProblemPoisson:

    def __init__(self, experiment) -> None:

        u = TrialFunction(experiment.V)
        v = TestFunction(experiment.V)

        f = Function(experiment.V)
        x = SpatialCoordinate(experiment.mesh)
        f = (8*pi*pi)*sin(x[0]*pi*2)*sin(x[1]*pi*2)

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

experiment = ExperimentPoisson()
problem = ProblemPoisson(experiment)
problem.solve()

# Create some fake data that is systematically different from the FEM solution.
# note that all parameters are on a log scale, so we take the true values
# and take the logarithm

# forcing covariance parameters (taken to be known)
sigma_f = np.log(2.e-2)
l_f = np.log(0.354)

# true model discrepancy parameters (need to be estimated)
rho = np.log(0.7)
sigma_eta = np.log(1.e-2)
l_eta = np.log(0.5)

# data statistical errors (taken to be known)
sigma_y = 2.e-3
datagrid = 6
ndata = datagrid**2

# statfem parameters
statfem_param = dict()
statfem_param['sigma_f'] = [sigma_f]
statfem_param['l_f'] = [l_f]
statfem_param['true_rho'] = rho
statfem_param['true_sigma_eta'] = sigma_eta
statfem_param['true_l_eta'] = l_eta
statfem_param['sigma_y'] = sigma_y
statfem_param['inference_mode'] = 'MAP'
statfem_param['stabilise'] = False

# create fake data on a grid
x_data = np.zeros((ndata, 2))
count = 0
for i in range(datagrid):
    for j in range(datagrid):
        x_data[count, 0] = float(i+1)/float(datagrid + 1)
        x_data[count, 1] = float(j+1)/float(datagrid + 1)
        count += 1

# fake data is the true FEM solution, scaled by the mismatch factor rho, with
# correlated errors added due to model/data discrepancy and uncorrelated measurement
# errors both added to the data
y = (np.exp(rho)*np.sin(2.*np.pi*x_data[:,0])*np.sin(2.*np.pi*x_data[:,1]) +
     np.random.multivariate_normal(mean = np.zeros(ndata), cov = sqexp(x_data, x_data, sigma_eta, l_eta)) +
     np.random.normal(scale = sigma_y, size = ndata))

statfem_problem = stat_fem.StatFEMProblem(problem, experiment, x_data, y, parameters=statfem_param)
statfem_problem.solve()
