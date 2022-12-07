import numpy as np
import json
import sys
sys.path.append("/home/darcones/firedrake/stat-fem")
import stat_fem
# from firedrake import *
from types import MethodType
sys.path.append("/home/darcones/FenicsConcrete")
import matplotlib.pyplot as plt
import fenicsX_concrete
from dolfinx.fem import Function, Constant
from dolfinx import io

try:
    import matplotlib.pyplot as plt
    makeplots = True
except ImportError:
    makeplots = False

def simple_lambda_(mesh, E, nu): #Lame's constant
    # return Constant(mesh, float(E * nu/((1 + E)*(1-2*nu))))
    return float(E * nu/((1 + E)*(1-2*nu)))

def simple_mu(mesh, E, nu):     #Lame's constant
    # return Constant(mesh, float(E/(2*(1+nu))))
    return float(E/(2*(1+nu)))

# Set up base FEM, which solves Poisson's equation on a square mesh
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

if __name__ == "__main__":
    makeplots = True
    nx = 33
    # Scaled variable

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

        plt.figure()
        plt.tripcolor(experiment.mesh.geometry.x[:,0], experiment.mesh.geometry.x[:,1],np.reshape(problem.displacement.vector.array, (-1,2))[:,0])
        plt.colorbar()
        plt.title('Original data displacement in x')
        plt.savefig("/home/darcones/Desktop/Figures/statFEM/GRF/original_displacement_x")

        plt.figure()
        plt.tripcolor(experiment.mesh.geometry.x[:,0], experiment.mesh.geometry.x[:,1],np.reshape(problem.displacement.vector.array, (-1,2))[:,1])
        plt.colorbar()
        plt.title('Original data displacement in y')
        plt.savefig("/home/darcones/Desktop/Figures/statFEM/GRF/original_displacement_x")

        plt.show()

    # A = assemble(problem.a, bcs = experiment.bcs)

    # Simplify field to be deterministic

    # problem.lambda_ = MethodType(simple_lambda_, problem)
    # problem.mu = MethodType(simple_mu, problem)
    problem.lambda_ = Constant(experiment.mesh, simple_lambda_(experiment.mesh, p.E+100, p.nu))
    problem.mu = Constant(experiment.mesh,simple_mu(experiment.mesh, p.E+100, p.nu))

    problem.define_weakform_problem()
    problem.solve()
    y_simple = [sensor.data[-1] for sensor_name,sensor in problem.sensors.items()]

    if  makeplots:
        difference = original_array_disp-np.reshape(problem.displacement.vector.array, (-1,2))
        norm_difference_unbiased = np.divide(difference , np.abs(np.reshape(problem.displacement.vector.array, (-1,2))))
        norm_difference_biased = np.divide(difference , np.abs(np.reshape(problem.displacement.vector.array, (-1,2)) + np.ones_like(difference)))
        plt.figure()
        plt.tripcolor(experiment.mesh.geometry.x[:,0], experiment.mesh.geometry.x[:,1],np.reshape(problem.displacement.vector.array, (-1,2))[:,0])
        plt.colorbar()
        plt.title('Model displacement in x')
        plt.savefig("/home/darcones/Desktop/Figures/statFEM/GRF/model_displacement_x")

        plt.figure()
        plt.tripcolor(experiment.mesh.geometry.x[:,0], experiment.mesh.geometry.x[:,1],np.reshape(problem.displacement.vector.array, (-1,2))[:,1])
        plt.colorbar()
        plt.title('Model displacement in y')
        plt.savefig("/home/darcones/Desktop/Figures/statFEM/GRF/model_displacement_y")

        plt.figure()
        plt.tripcolor(experiment.mesh.geometry.x[:,0], experiment.mesh.geometry.x[:,1],difference[:,0])
        plt.colorbar()
        plt.title('Model error in x')
        plt.savefig("/home/darcones/Desktop/Figures/statFEM/GRF/model_error_x")

        plt.figure()
        plt.tripcolor(experiment.mesh.geometry.x[:,0], experiment.mesh.geometry.x[:,1],difference[:,1])
        plt.colorbar()
        plt.title('Model error in y')
        plt.savefig("/home/darcones/Desktop/Figures/statFEM/GRF/model_error_y")

        plt.figure()
        plt.tripcolor(experiment.mesh.geometry.x[:,0], experiment.mesh.geometry.x[:,1],norm_difference_biased[:,0])
        plt.colorbar()
        plt.title('Normalized model error in x')
        plt.savefig("/home/darcones/Desktop/Figures/statFEM/GRF/normalized_model_error_x")

        plt.figure()
        plt.tripcolor(experiment.mesh.geometry.x[:,0], experiment.mesh.geometry.x[:,1],norm_difference_unbiased[:,1])
        plt.colorbar()
        plt.title('Normalized model error in y')
        plt.savefig("/home/darcones/Desktop/Figures/statFEM/GRF/normalized_model_error_y")
        # plt.show()

    with io.XDMFFile(experiment.mesh.comm, "deformation_stochastic_slab.xdmf", "w") as xdmf:
        xdmf.write_mesh(experiment.mesh)
        problem.displacement.name = "Deformation prior"
        xdmf.write_function(problem.displacement)
        
        statfem_problem = stat_fem.StatFEMProblem(problem, experiment, x_data, y, parameters=statfem_param, makeplots = False)
        statfem_problem.solve()

        statfem_problem.mu_mesh.name = "Deformation posterior"
        xdmf.write_function(statfem_problem.mu_mesh)

    keys = {'rho', 'sigma_d', 'l_d'}
    params_dic = {k: np.exp(v) for k, v in zip(keys, statfem_problem.ls.params)}

    with open("statfem_parameters.json", "w") as outfile:
        json.dump(params_dic, outfile)

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
                    np.reshape(statfem_problem.muy.x.array, (-1,2))[:,1])
        plt.colorbar(label="Displacement posterior",pad=0.1)
        plt.scatter(x_data[:,0], x_data[:,1], c = np.diag(statfem_problem.Cuy)[1::2], cmap="Greys_r")
        plt.colorbar(label="Covariance $C_{u|y}$ at y")
        plt.title("Posterior FEM solution and uncertainty (at mesh)")
        plt.savefig("/home/darcones/Desktop/Figures/statFEM/GRF/posterior_vertical_at_mesh")

        plt.figure()
        plt.scatter(x_data[:,0], x_data[:,1], c = statfem_problem.muy2[:,1])
        plt.colorbar(label="Displacement $u|y$ at y")
        plt.title("Posterior FEM solution and uncertainty (at data)")
        plt.savefig("/home/darcones/Desktop/Figures/statFEM/GRF/posterior_vertical_at_data")

        plt.figure()
        plt.plot(statfem_problem.muy2[:,1], 'b-', label = "u|y")
        plt.plot(statfem_problem.muy2[:,1]+1.96*np.diag(statfem_problem.Cuy)[1::2], 'b-', label = r"u+3$\sigma$",alpha = 0.5)
        plt.plot(statfem_problem.muy2[:,1]-1.96*np.diag(statfem_problem.Cuy)[1::2], 'b-', label = r"u-3$\sigma$",alpha = 0.5)
        plt.plot(statfem_problem.muy2[:,1]+1.96*(np.diag(statfem_problem.Cuy)[1::2]+np.exp(statfem_problem.ls.params[1])), 'g-', label = r"u+3$\sigma$+$d$",alpha = 0.5)
        plt.plot(statfem_problem.muy2[:,1]-1.96*(np.diag(statfem_problem.Cuy)[1::2]+np.exp(statfem_problem.ls.params[1])), 'g-', label = r"u-3$\sigma$+$d$",alpha = 0.5)
        plt.plot(np.array(y)[:,1], 'ro',  markersize = 2, label = "True y")
        plt.plot(np.array(y_simple)[:,1], 'k+', label = "Simple y")
        plt.ylabel("Displacement [m]")
        plt.xlabel("Data point")
        plt.title('Posterior vertical displacement at sensors')
        plt.legend()
        plt.savefig("/home/darcones/Desktop/Figures/statFEM/GRF/posterior_vertical_at_sensors")

        plt.figure()
        plt.plot(statfem_problem.muy2[:,0], 'b-', label = "u|y")
        plt.plot(statfem_problem.muy2[:,0]+1.96*np.diag(statfem_problem.Cuy)[::2], 'b-', label = r"u+3$\sigma$",alpha = 0.5)
        plt.plot(statfem_problem.muy2[:,0]-1.96*np.diag(statfem_problem.Cuy)[::2], 'b-', label = r"u-3$\sigma$",alpha = 0.5)
        plt.plot(statfem_problem.muy2[:,0]+1.96*(np.diag(statfem_problem.Cuy)[::2]+np.exp(statfem_problem.ls.params[1])), 'g-', label = r"u+3$\sigma$+$d$",alpha = 0.5)
        plt.plot(statfem_problem.muy2[:,0]-1.96*(np.diag(statfem_problem.Cuy)[::2]+np.exp(statfem_problem.ls.params[1])), 'g-', label = r"u-3$\sigma$+$d$",alpha = 0.5)
        plt.plot(np.array(y)[:,0], 'ro', markersize = 2, label = "y")
        plt.plot(np.array(y_simple)[:,0], 'k+', label = "Simple y")
        plt.title('Posterior horizontal displacement at sensors')
        plt.ylabel("Displacement [m]")
        plt.xlabel("Data point")
        plt.legend()
        plt.savefig("/home/darcones/Desktop/Figures/statFEM/GRF/posterior_horizontal_at_sensors")
        # plt.show()

        plt.figure()
        plt.plot(np.array(y)[:,1]-np.array(y_simple)[:,1], 'b--', label='Prior')
        plt.plot(np.array(y)[:,1]-statfem_problem.muy2[:,1], 'b-', label='Posterior')
        plt.ylabel("Displacement [m]")
        plt.xlabel("Data point")
        plt.title('Model mismatch for vertical displacement')
        plt.legend()
        plt.savefig("/home/darcones/Desktop/Figures/statFEM/GRF/model_mismatch_vertical")

        plt.figure()
        plt.plot(np.array(y)[:,0]-np.array(y_simple)[:,0], 'g--', label='Prior')
        plt.plot(np.array(y)[:,0]-statfem_problem.muy2[:,0], 'g-', label='Posterior')
        plt.ylabel("Displacement [m]")
        plt.xlabel("Data point")
        plt.title('Model mismatch for horizontal displacement')
        plt.legend()
        plt.savefig("/home/darcones/Desktop/Figures/statFEM/GRF/model_mismatch_horizontal")

        plt.figure()
        plt.plot(statfem_problem.muy2[:,1]-np.array(y_simple)[:,1], 'b-')
        plt.ylabel("Displacement [m]")
        plt.xlabel("Data point")
        plt.title('Absolute vertical displacement model update at data')
        plt.savefig("/home/darcones/Desktop/Figures/statFEM/GRF/absolute_update_vertical")

        plt.figure()
        plt.plot(statfem_problem.muy2[:,0]-np.array(y_simple)[:,0], 'g-')
        plt.ylabel("Displacement [m]")
        plt.xlabel("Data point")
        plt.title('Absolute horizontal displacement model update at data')
        plt.savefig("/home/darcones/Desktop/Figures/statFEM/GRF/absolute_update_horizontal")

        plt.figure()
        plt.plot(np.abs(np.array(y_simple)[:,1]-statfem_problem.muy2[:,1])/np.abs(statfem_problem.muy2[:,1])*100, 'b-')
        plt.ylabel("Displacement [%]")
        plt.xlabel("Data point")
        plt.title('Relative vertical displacement model update at data')
        plt.savefig("/home/darcones/Desktop/Figures/statFEM/GRF/relative_update_vertical")

        plt.figure()
        plt.plot(np.abs(np.array(y_simple)[:,0]-statfem_problem.muy2[:,0])/np.abs(statfem_problem.muy2[:,0])*100, 'g-')
        plt.ylabel("Displacement [%]")
        plt.xlabel("Data point")
        plt.title('Relative horizontal displacement model update at data')
        plt.savefig("/home/darcones/Desktop/Figures/statFEM/GRF/relative_update_horizontal")

        original_error = np.array(y)-np.array(y_simple)
        original_error_relative = np.divide(original_error, statfem_problem.muy2)
        final_error = np.array(y)-statfem_problem.muy2
        final_error_relative = np.divide(final_error, statfem_problem.muy2)
        absolute_improvement = np.abs(final_error)-np.abs(original_error)
        relative_improvement = absolute_improvement/np.abs(original_error)
        plt.figure()
        plt.plot(absolute_improvement[:,1], 'b-')
        plt.ylabel("Error [m]")
        plt.xlabel("Data point")
        plt.title('Absolute improvement vertical displacement model update at data')
        plt.savefig("/home/darcones/Desktop/Figures/statFEM/GRF/absolute_improvement_vertical")

        plt.figure()
        plt.plot(absolute_improvement[:,0], 'g-')
        plt.ylabel("Error [m]")
        plt.xlabel("Data point")
        plt.title('Absolute improvement horizontal displacement model update at data')
        plt.savefig("/home/darcones/Desktop/Figures/statFEM/GRF/absolute_improvement_horizontal")

        plt.figure()
        plt.plot(relative_improvement[:,1]*100, 'b-')
        plt.ylabel("Error [%]")
        plt.xlabel("Data point")
        plt.title('Relative improvement vertical displacement model update at data')
        plt.savefig("/home/darcones/Desktop/Figures/statFEM/GRF/relative_improvement_vertical")

        plt.figure()
        plt.plot(relative_improvement[:,0]*100, 'g-')
        plt.ylabel("Error [%]")
        plt.xlabel("Data point")
        plt.title('Relative improvement horizontal displacement model update at data')
        plt.savefig("/home/darcones/Desktop/Figures/statFEM/GRF/relative_improvement_horizontal")

        print(f"MSE in vertical (absolute) for original is: {np.square(original_error[:,1]).mean()}")
        print(f"MSE in vertical (absolute) for updated is: {np.square(final_error[:,1]).mean()}")
        mse_vertical_absolute_1 =-(np.square(final_error[:,1]).mean()/np.square(original_error[:,1]).mean()-1)*100
        print(f"Improvement in MSE for vertical (absolute) displacement is {mse_vertical_absolute_1} %\n")
        
        print(f"MSE in horizontal (absolute) for original is: {np.square(original_error[:,0]).mean()}")
        print(f"MSE in horizontal (absolute) for updated is: {np.square(final_error[:,0]).mean()}")
        mse_horizontal_absolute_1 =-(np.square(final_error[:,0]).mean()/np.square(original_error[:,0]).mean()-1)*100
        print(f"Improvement in MSE for horizontal (absolute) displacement is {mse_horizontal_absolute_1} %\n")
        
        print(f"MSE in vertical (relative) for original is: {np.square(original_error_relative[:,1]).mean()}")
        print(f"MSE in vertical (relative) for updated is: {np.square(final_error_relative[:,1]).mean()}")
        mse_vertical_relative_1 = -(np.square(final_error_relative[:,1]).mean()/np.square(original_error_relative[:,1]).mean()-1)*100
        print(f"Improvement in MSE for vertical (relative) displacement is {mse_vertical_relative_1} %\n")
        
        print(f"MSE in horizontal (relative) for original is: {np.square(original_error_relative[:,0]).mean()}")
        print(f"MSE in horizontal (relative) for updated is: {np.square(final_error_relative[:,0]).mean()}")
        mse_horizontal_relative_1 = -(np.square(final_error_relative[:,0]).mean()/np.square(original_error_relative[:,0]).mean()-1)*100
        print(f"Improvement in MSE for horizontal (relative) displacement is {mse_horizontal_relative_1} %\n")

        # plt.show()

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

    p.E = p.E + 10.0
    problem.define_variational_problem()
    problem.solve()
    y_simple = [sensor.data[-1] for sensor_name,sensor in problem.sensors.items()]

    statfem_problem = stat_fem.StatFEMProblem(problem, experiment, x_data, y, parameters=statfem_param, makeplots = False)
    statfem_problem.solve()

    original_error = np.array(y)-np.array(y_simple)
    original_error_relative = np.divide(original_error, statfem_problem.muy2)
    final_error = np.array(y)-statfem_problem.muy2
    final_error_relative = np.divide(final_error, statfem_problem.muy2)
    absolute_improvement = np.abs(final_error)-np.abs(original_error)
    relative_improvement = absolute_improvement/np.abs(original_error)

    print(f"MSE in vertical (absolute) for original is: {np.square(original_error[:,1]).mean()}")
    print(f"MSE in vertical (absolute) for updated is: {np.square(final_error[:,1]).mean()}")
    mse_vertical_absolute_2 =-(np.square(final_error[:,1]).mean()/np.square(original_error[:,1]).mean()-1)*100
    print(f"Improvement in MSE for vertical (absolute) displacement is {mse_vertical_absolute_2} %\n")

    print(f"MSE in horizontal (absolute) for original is: {np.square(original_error[:,0]).mean()}")
    print(f"MSE in horizontal (absolute) for updated is: {np.square(final_error[:,0]).mean()}")
    mse_horizontal_absolute_2 =-(np.square(final_error[:,0]).mean()/np.square(original_error[:,0]).mean()-1)*100
    print(f"Improvement in MSE for horizontal (absolute) displacement is {mse_horizontal_absolute_2} %\n")

    print(f"MSE in vertical (relative) for original is: {np.square(original_error_relative[:,1]).mean()}")
    print(f"MSE in vertical (relative) for updated is: {np.square(final_error_relative[:,1]).mean()}")
    mse_vertical_relative_2 = -(np.square(final_error_relative[:,1]).mean()/np.square(original_error_relative[:,1]).mean()-1)*100
    print(f"Improvement in MSE for vertical (relative) displacement is {mse_vertical_relative_2} %\n")

    print(f"MSE in horizontal (relative) for original is: {np.square(original_error_relative[:,0]).mean()}")
    print(f"MSE in horizontal (relative) for updated is: {np.square(final_error_relative[:,0]).mean()}")
    mse_horizontal_relative_2 = -(np.square(final_error_relative[:,0]).mean()/np.square(original_error_relative[:,0]).mean()-1)*100
    print(f"Improvement in MSE for horizontal (relative) displacement is {mse_horizontal_relative_2} %\n")
