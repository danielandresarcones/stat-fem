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
p['num_elements_length'] = 20
p['num_elements_breadth'] = 10
p['dim'] = 2
#displacement = -3

# data statistical errors (taken to be known)
sigma_y = 2.e-3
datagrid_x = 6
datagrid_y = 3
ndata = datagrid_x*datagrid_y

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
# A = assemble(problem.a, bcs = experiment.bcs)

# Simplify field to be deterministic
def simple_lambda_(self): #Lame's constant
        return (self.p.E.mean * self.p.nu.mean)/((1 + self.p.E.mean)*(1-2*self.p.nu.mean))

def simple_mu(self):     #Lame's constant
    return self.p.E.mean/(2*(1+self.p.nu.mean))

problem.lambda_ = MethodType(simple_lambda_, problem)
problem.mu = MethodType(simple_mu, problem)


statfem_problem = stat_fem.StatFEMProblem(problem, experiment, x_data, y)
statfem_problem.solve()