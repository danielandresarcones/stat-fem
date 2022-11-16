import numpy as np
import sys
sys.path.append("/home/darcones/firedrake/stat-fem")
import stat_fem
# from firedrake import *
from types import MethodType
sys.path.append("/home/darcones/FenicsConcrete")
import matplotlib.pyplot as plt
import fenicsX_concrete
from dolfinx.fem import Function
from dolfinx import io

# local imports (problem definition)
from probeye.definition.inverse_problem import InverseProblem
from probeye.definition.forward_model import ForwardModelBase
from probeye.definition.distribution import Normal, Uniform, LogNormal
from probeye.definition.sensor import Sensor
from probeye.definition.likelihood_model import GaussianLikelihoodModel

# local imports (problem solving)
from probeye.inference.scipy.solver import MaxLikelihoodSolver
from probeye.inference.emcee.solver import EmceeSolver
from probeye.definition.correlation_model import PrescribedCovModel

# local imports (inference data post-processing)
from probeye.postprocessing.sampling_plots import create_pair_plot
from probeye.postprocessing.sampling_plots import create_posterior_plot
from probeye.postprocessing.sampling_plots import create_trace_plot

try:
    import matplotlib.pyplot as plt
    makeplots = True
except ImportError:
    makeplots = False

def simple_lambda_(E, nu): #Lame's constant
    return lambda x: E * nu/((1 + E)*(1-2*nu))

def simple_mu(E, nu):     #Lame's constant
    return lambda x: E/(2*(1+nu))

class StatFEMInferenceModel(ForwardModelBase):

    def __init__(self, name: str, *args, **kwargs):
        super().__init__(name, *args, **kwargs)
        self.statfem_problem = statfem_problem


    def interface(self):
        self.parameters = ["E", "nu"]
        self.input_sensors = []
        self.output_sensors = Sensor("y", std_model="model_variance")

    def response(self, inp: dict) -> dict:
        E = inp["E"]
        nu = inp["nu"]

        # Update parameters
        self.statfem_problem.problem.lambda_ = MethodType(simple_lambda_(E, nu), problem )
        self.statfem_problem.problem.mu = MethodType(simple_mu(E, nu), problem)

        # Update formulation and solve linear problem
        self.statfem_problem.problem.define_variational_problem()
        self.statfem_problem.solve_lp()
   
        return{"y": np.exp(self.statfem_problem.ls.params[0])*self.statfem_problem.mu}


if __name__ == "__main__":
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
    p['E'] = 160
    p['nu'] = 0.3
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

    if makeplots:

        original_array_disp = np.reshape(problem.displacement.vector.array, (-1,2)).copy()
        plt.figure()
        plt.tripcolor(experiment.mesh.geometry.x[:,0], experiment.mesh.geometry.x[:,1],problem.p.E.field.vector.array)
        plt.colorbar()
        plt.title('Original data E field')
        plt.savefig("/home/darcones/Desktop/Figures/statFEM/GRF/original_E")

        plt.figure()
        plt.tripcolor(experiment.mesh.geometry.x[:,0], experiment.mesh.geometry.x[:,1],problem.p.nu.field.vector.array)
        plt.colorbar()
        plt.title('Original data nu field')
        plt.savefig("/home/darcones/Desktop/Figures/statFEM/GRF/original_nu")
        # plt.show()

    # A = assemble(problem.a, bcs = experiment.bcs)

    # Simplify field to be deterministic
    

    problem.lambda_ = MethodType(simple_lambda_(p.E, p.nu), problem )
    problem.mu = MethodType(simple_mu(p.E, p.nu), problem )

    problem.define_variational_problem()
    problem.solve()
    y_simple = [sensor.data[-1] for sensor_name,sensor in problem.sensors.items()]



    with io.XDMFFile(experiment.mesh.comm, "deformation_stochastic_slab.xdmf", "w") as xdmf:
        xdmf.write_mesh(experiment.mesh)
        problem.displacement.name = "Deformation prior"
        xdmf.write_function(problem.displacement)
        
        statfem_problem = stat_fem.StatFEMProblem(problem, experiment, x_data, y, parameters=statfem_param, makeplots = False)
        statfem_problem.solve()

        statfem_problem.mu_mesh.name = "Deformation posterior"
        xdmf.write_function(statfem_problem.mu_mesh)

    forward_model  = StatFEMInferenceModel("Parameter_inference_model", statfem_problem = statfem_problem)

    inverse_problem = InverseProblem("Linear regression with Gaussian noise", print_header=False)

    # add the problem's parameters
    inverse_problem.add_parameter(
        "E",
        tex="$E$",
        info="Young's modulus",
        # domain="[10, 1000)",
        # prior=LogNormal(mean=float(np.log(90)), std=0.5),
        prior=Normal(mean=120, std=15),
    )
    inverse_problem.add_parameter(
        "nu",
        tex="$\\nu$",
        info="Poisson's ration",
        # prior=LogNormal(mean=float(np.log(0.18)), std=0.05),
        # domain = "(0, 0.5)",
        # prior=Normal(mean=0.05, std=0.2),
        value = 0.3,
    )

    post_cov = statfem_problem.ls.data.calc_K_plus_sigma(statfem_problem.ls.params[1:])

    inverse_problem.add_parameter(
        "model_variance",
        value = np.diag(post_cov),
        tex=r"Diagonal $C_d+C_e$",
        info="Standard deviation from the Gaussian model",
    )

    inverse_problem.add_parameter(
        "cov",
        value = post_cov,
        tex=r"$C_d+C_e$",
        info="Covariance matrix of the Gaussian model",
    )

    inverse_problem.add_experiment(
        name="TestSeries_1",
        sensor_data={"y": np.reshape(y, (-1,))},
    )

    # forward model
    inverse_problem.add_forward_model(forward_model, experiments="TestSeries_1")
    inverse_problem.forward_models["Parameter_inference_model"].sensors_share_std_model = False # Necessary to define covariance at each sample point

    # likelihood model
    inverse_problem.add_likelihood_model(
        GaussianLikelihoodModel(experiment_name="TestSeries_1", model_error="additive", correlation=PrescribedCovModel(cov = 'cov'))
    )

    inverse_problem.info(print_header=True)

    emcee_solver = EmceeSolver(inverse_problem, show_progress=True)
    inference_data = emcee_solver.run(n_steps=60, n_initial_steps=10,n_walkers = 10)

    true_values = { 'E': 100, 'nu':0.2}
    pair_plot_array = create_pair_plot(
        inference_data,
        emcee_solver.problem,
        true_values=true_values,
        focus_on_posterior=True,
        show_legends=True,
        title="Sampling results from emcee-Solver (pair plot)",
    )

    post_plot_array = create_posterior_plot(
        inference_data,
        emcee_solver.problem,
        true_values=true_values,
        title="Sampling results from emcee-Solver (posterior plot)",
    )

    trace_plot_array = create_trace_plot(
        inference_data,
        emcee_solver.problem,
        title="Sampling results from emcee-Solver (trace plot)",
    )

    # Update parameters
    # statfem_problem.problem.lambda_ = MethodType(simple_lambda_(emcee_solver.summary['mean']['E'], emcee_solver.summary['mean']['nu']), problem )
    # statfem_problem.problem.mu = MethodType(simple_mu(emcee_solver.summary['mean']['E'], emcee_solver.summary['mean']['nu']), problem)
    statfem_problem.problem.lambda_ = MethodType(simple_lambda_(emcee_solver.summary['mean']['E'], p["nu"]), problem )
    statfem_problem.problem.mu = MethodType(simple_mu(emcee_solver.summary['mean']['E'], p["nu"]), problem)

    # Update formulation and solve linear problem
    statfem_problem.problem.define_variational_problem()
    statfem_problem.solve_lp()

    plt.figure()
    plt.plot(statfem_problem.muy2[:,1], 'b-', label = "u|y")
    plt.plot(statfem_problem.muy2[:,1]+1.96*np.diag(statfem_problem.Cuy)[1::2], 'b-', label = "u+3sigma",alpha = 0.5)
    plt.plot(statfem_problem.muy2[:,1]-1.96*np.diag(statfem_problem.Cuy)[1::2], 'b-', label = "u-3sigma",alpha = 0.5)
    plt.plot(statfem_problem.mu[1::2],'g-', label = "u updated parameters")
    plt.plot(np.array(y)[:,1], 'ro',  markersize = 2, label = "True y")
    plt.plot(np.array(y_simple)[:,1], 'k+', label = "Simple y")
    plt.ylabel("Displacement [m]")
    plt.xlabel("Data point")
    plt.title('Posterior vertical displacement at sensors')
    # plt.savefig("/home/darcones/Desktop/Figures/statFEM/GRF/posterior_vertical_at_sensors")
    plt.legend()

    plt.figure()
    plt.plot(statfem_problem.muy2[:,0], 'b-', label = "u|y")
    plt.plot(statfem_problem.muy2[:,0]+1.96*np.diag(statfem_problem.Cuy)[::2], 'b-', label = "u+3sigma",alpha = 0.5)
    plt.plot(statfem_problem.muy2[:,0]-1.96*np.diag(statfem_problem.Cuy)[::2], 'b-', label = "u-3sigma",alpha = 0.5)
    plt.plot(statfem_problem.mu[::2],'g-', label = "u updated parameters")
    plt.plot(np.array(y)[:,0], 'ro', markersize = 2, label = "y")
    plt.plot(np.array(y_simple)[:,0], 'k+', label = "Simple y")
    plt.title('Posterior horizontal displacement at sensors')
    plt.ylabel("Displacement [m]")
    plt.xlabel("Data point")
    plt.legend()
    # plt.savefig("/home/darcones/Desktop/Figures/statFEM/GRF/posterior_horizontal_at_sensors")
    plt.show()
