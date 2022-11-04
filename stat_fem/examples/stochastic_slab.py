import numpy as np
import sys
sys.path.append("/home/darcones/firedrake/stat-fem")
import stat_fem
import matplotlib.pyplot as plt
# from firedrake import *
from types import MethodType
sys.path.append("/home/darcones/FenicsConcrete")
import fenicsX_concrete
from dolfinx.fem import Function
from dolfinx import io

try:
    import matplotlib.pyplot as plt
    makeplots = True
except ImportError:
    makeplots = False

# Set up base FEM, which solves Poisson's equation on a square mesh

makeplots = True
nx = 33
# Scaled variable


def simple_setup(p, sensors):
    parameters = fenicsX_concrete.Parameters()  # using the current default values

    #parameters['log_level'] = 'WARNING'
    parameters['bc_setting'] = 'free'
    parameters['mesh_density'] = 10

    parameters = parameters + p

    experiment = fenicsX_concrete.concreteSlabExperiment(parameters)         # Specifies the domain, discretises it and apply Dirichlet BCs

    problem = fenicsX_concrete.LinearElasticity(experiment, parameters)      # Specifies the material law and weak forms.
    #print(help(fenics_concrete.LinearElasticity))

    for sensor in sensors:
        problem.add_sensor(sensor)

    problem.solve()

    return experiment, problem



p = fenicsX_concrete.Parameters()  # using the current default values
p['E'] = 100
p['nu'] = 0.2
p['length'] = 1
p['breadth'] = 0.2
p['num_elements_length'] = 10
p['num_elements_breadth'] = 5
p['dim'] = 2
#displacement = -3

# data statistical errors (taken to be known)
sigma_y = 2.e-3
datagrid_x = 6
datagrid_y = 3
ndata = datagrid_x*datagrid_y

# statfem parameters
statfem_param = dict()
statfem_param['sigma_f'] = [np.log(0.5), np.log(0.5)]
statfem_param['l_f'] = [np.log(0.353), np.log(0.353)]
statfem_param['sigma_y'] = sigma_y
statfem_param['inference_mode'] = 'MCMC'
statfem_param['stabilise'] = True

# create fake data on a grid
x_data = np.zeros((ndata, 2))
count = 0
sensors = []
for i in range(datagrid_x):
    for j in range(datagrid_y):
        x_data[count, 0] = float(i+1)/float(datagrid_x + 1) * p['length']
        x_data[count, 1] = float(j+1)/float(datagrid_y + 1) * p['breadth']
        sensors.append(fenicsX_concrete.sensors.DisplacementSensor(np.array([[x_data[count, 0], x_data[count, 1], 0]])))
        count += 1

#print(sensor.name)
experiment, problem = simple_setup(p, sensors)

y = [sensor.data[-1] for sensor_name,sensor in problem.sensors.items()]

if not makeplots:

    original_array_disp = np.reshape(problem.displacement.vector.array, (-1,2)).copy()
    plt.figure()
    plt.tripcolor(experiment.mesh.geometry.x[:,0], experiment.mesh.geometry.x[:,1],problem.p.E.field.vector.array)
    plt.colorbar()
    plt.title('Original data E field')

    plt.figure()
    plt.tripcolor(experiment.mesh.geometry.x[:,0], experiment.mesh.geometry.x[:,1],problem.p.nu.field.vector.array)
    plt.colorbar()
    plt.title('Original data nu field')

    plt.figure()
    plt.tripcolor(experiment.mesh.geometry.x[:,0], experiment.mesh.geometry.x[:,1],np.reshape(problem.displacement.vector.array, (-1,2))[:,0])
    plt.colorbar()
    plt.title('Original data displacement in x')

    plt.figure()
    plt.tripcolor(experiment.mesh.geometry.x[:,0], experiment.mesh.geometry.x[:,1],np.reshape(problem.displacement.vector.array, (-1,2))[:,1])
    plt.colorbar()
    plt.title('Original data displacement in y')

    # plt.show()

# A = assemble(problem.a, bcs = experiment.bcs)

# Simplify field to be deterministic
def simple_lambda_(self): #Lame's constant
    return p.E * p.nu/((1 + p.E)*(1-2*p.nu))

def simple_mu(self):     #Lame's constant
    return p.E/(2*(1+p.nu))

problem.lambda_ = MethodType(simple_lambda_, problem)
problem.mu = MethodType(simple_mu, problem)

problem.define_variational_problem()
problem.solve()

if  not makeplots:
    difference = original_array_disp-np.reshape(problem.displacement.vector.array, (-1,2))
    norm_difference_unbiased = np.divide(difference , np.abs(np.reshape(problem.displacement.vector.array, (-1,2))))
    norm_difference_biased = np.divide(difference , np.abs(np.reshape(problem.displacement.vector.array, (-1,2)) + np.ones_like(difference)))
    plt.figure()
    plt.tripcolor(experiment.mesh.geometry.x[:,0], experiment.mesh.geometry.x[:,1],np.reshape(problem.displacement.vector.array, (-1,2))[:,0])
    plt.colorbar()
    plt.title('Model displacement in x')

    plt.figure()
    plt.tripcolor(experiment.mesh.geometry.x[:,0], experiment.mesh.geometry.x[:,1],np.reshape(problem.displacement.vector.array, (-1,2))[:,1])
    plt.colorbar()
    plt.title('Model displacement in y')

    plt.figure()
    plt.tripcolor(experiment.mesh.geometry.x[:,0], experiment.mesh.geometry.x[:,1],difference[:,0])
    plt.colorbar()
    plt.title('Model error in x')

    plt.figure()
    plt.tripcolor(experiment.mesh.geometry.x[:,0], experiment.mesh.geometry.x[:,1],difference[:,1])
    plt.colorbar()
    plt.title('Model error in y')

    plt.figure()
    plt.tripcolor(experiment.mesh.geometry.x[:,0], experiment.mesh.geometry.x[:,1],norm_difference_biased[:,0])
    plt.colorbar()
    plt.title('Normalized model error in x')

    plt.figure()
    plt.tripcolor(experiment.mesh.geometry.x[:,0], experiment.mesh.geometry.x[:,1],norm_difference_unbiased[:,1])
    plt.colorbar()
    plt.title('Normalized model error in y')
    plt.show()

with io.XDMFFile(experiment.mesh.comm, "deformation_stochastic_slab.xdmf", "w") as xdmf:
    xdmf.write_mesh(experiment.mesh)
    problem.displacement.name = "Deformation prior"
    xdmf.write_function(problem.displacement)
    
    statfem_problem = stat_fem.StatFEMProblem(problem, experiment, x_data, y, parameters=statfem_param, makeplots = False)
    statfem_problem.solve()

    statfem_problem.mu_mesh.name = "Deformation posterior"
    xdmf.write_function(statfem_problem.mu_mesh)

# # plot priors 
# if makeplots:
#     plt.figure()
#     plt.plot(x_data[:,0],statfem_problem.mu[:,1],'o-',markersize = 2,label = "u")
#     plt.fill_between(x_data[:,0],statfem_problem.mu[:,1]+1.96*np.diag(statfem_problem.Cu)[1::2], statfem_problem.mu[:,1]-1.96*np.diag(statfem_problem.Cu)[1::2], label = "u 95 confidence",alpha = 0.5)
#     plt.plot(x_data[:,0],z_mean, '+-', label = "True z")
#     plt.fill_between(x_data[:,0],z_mean+1.96*np.sqrt(np.diag(z_cov)),z_mean-1.96*np.sqrt(np.diag(z_cov)), label = "True z 95 confidence", alpha = 0.5)
#     plt.plot(x_data[:,0],y[:,1], '+', label = "data")
#     plt.xlabel("x [m]")
#     plt.ylabel("y")
#     plt.legend()
#     plt.title("Prior solutions")

if makeplots:
    plt.figure()
    plt.tripcolor(experiment.mesh.geometry.x[:,0], experiment.mesh.geometry.x[:,1],
                  np.reshape(statfem_problem.muy.vector, (-1,2))[:,1])
    plt.colorbar(label="Displacement posterior",pad=0.1)
    plt.scatter(x_data[:,0], x_data[:,1], c = np.diag(statfem_problem.Cuy)[1::2], cmap="Greys_r")
    plt.colorbar(label="Covariance $C_{u|y}$ at y")
    plt.title("Posterior FEM solution and uncertainty")
    plt.show()

# if makeplots:
#     plt.figure()
#     plt.plot(x_data[:,0],statfem_problem.muy2[:,1],'o-',markersize = 0, label = "u posterior scaled")
#     plt.fill_between(x_data[:,0],statfem_problem.muy2[:,1]+1.96*np.sqrt(np.diag(statfem_problem.Cuy)[1::2]), statfem_problem.muy2[:,1]-1.96*np.sqrt(np.diag(statfem_problem.Cuy)[1::2]), label = "u scaled 95 confidence", alpha = 0.5)
#     plt.plot(x_data[:,0],statfem_problem.mu_z2[:,1],'o-',markersize = 2,label = "y from FEM (z+noise) posterior")
#     plt.fill_between(x_data[:,0],statfem_problem.mu_z2[:,1]+1.96*np.sqrt(np.diag(statfem_problem.Cu_z2)[1::2]), statfem_problem.mu_z2[:,1]-1.96*np.sqrt(np.diag(statfem_problem.Cu_z2)[1::2]), label = "y from FEM (z+noise) 95 confidence", alpha = 0.5)
#     plt.plot(x_data[:,0],y[:,1], '+', label = "data")
#     plt.title("Posterior solutions")
#     plt.xlabel("x [m]")
#     plt.ylabel("y")
#     plt.legend()
#     plt.show()