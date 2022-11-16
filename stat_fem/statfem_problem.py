import numpy as np
import sys

sys.path.append("/home/darcones/firedrake/stat-fem") #TODO: Adjust for relative pathing
import stat_fem
import matplotlib.pyplot as plt
from scipy.linalg import cho_factor, cho_solve
from scipy.linalg import LinAlgError
# from firedrake import *
from types import MethodType
sys.path.append("/home/darcones/FenicsConcrete")
import fenicsX_concrete
from dolfinx.fem import Function, form
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, apply_lifting, set_bc
from petsc4py import PETSc
from mpi4py import MPI

try:
    import matplotlib.pyplot as plt
    makeplots = True
except ImportError:
    makeplots = False

class StatFEMProblem:

    def __init__(self, problem, experiment, data_coords, data_values, parameters: dict = None, makeplots = True):

        self._get_default_parameters()
        if not parameters is None:
            for key,value in parameters.items():
                self.parameters[key] = value

        self.solved = False
        self.problem = problem
        self.experiment = experiment
        self.data_coords = data_coords
        self.data_values = data_values
        self.makeplots = makeplots
        self.dim = len(data_coords[0])

        self.V = self.experiment.V
        self.n_data = len(self.data_coords)

        # Compute and assemble forcing covariance matrix using known correlated errors in forcing

        self.G = stat_fem.ForcingCovariance(self.V, self.parameters['sigma_f'], self.parameters['l_f'])
        self.G.assemble()

        self.lsolver = PETSc.KSP().create(MPI.COMM_WORLD)

        # combine data into an observational data object using known locations, observations,
        # and known statistical errors

        self.obs_data = stat_fem.ObsData(self.data_coords, self.data_values, self.parameters['sigma_y'])

    def solve(self):

        # Begin stat-fem solution

        true_values = np.array([self.parameters['true_rho'], self.parameters['true_sigma_eta'], self.parameters['true_l_eta']])

        # assemble matrices if not present
        if not hasattr(self.problem, 'A'):
            self.problem.A = assemble_matrix(form(self.problem.a), bcs = self.problem.experiment.bcs)
            self.problem.A.assemble()

        if not hasattr(self.problem, 'b'):
            self.problem.b = assemble_vector(form(self.problem.L))
            apply_lifting(self.problem.b, [form(self.problem.a)], bcs=[self.problem.experiment.bcs])
            self.problem.b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
            set_bc(self.problem.b, self.problem.experiment.bcs)

        if self.parameters['inference_mode'] == 'MAP':
            self.ls = stat_fem.estimate_params_MAP(self.problem, self.G, self.obs_data, stabilise = self.parameters['stabilise'])
        elif self.parameters['inference_mode'] == 'MCMC':
            self.ls, self.samples = stat_fem.estimate_params_MCMC(self.problem, self.G, self.obs_data, stabilise = self.parameters['stabilise'])
            if self.makeplots:
                figure, axes = plt.subplots(3)
                figure.suptitle("MCMC 20000 samples")
                names = [r"$\rho$",r"$\sigma_d$",r"$l_d$"]
                p_names = [r"$p(\rho)$",r"$p(\sigma_d$)",r"p($l_d$)"]
                # ranges =  [[0,1.5],[0,0.3],[0,0.2]]
                for i in range(3):
                    # axes[i].hist(np.exp(self.samples[:, i]), 100, range = ranges[i] ,color='k', histtype="step")
                    axes[i].hist(np.exp(self.samples[:, i]), 100, color='k', histtype="step")
                    axes[i].set(xlabel = names[i], ylabel = p_names[i])
                    axes[i].axvline(x=np.exp(self.ls.params[i]), c='b',linestyle = '-', label = "Estimate")
                    axes[i].axvline(x=np.exp(true_values[i]), c='r',linestyle = '--', label = "True")
                    axes[i].legend()
                axes[2].set(xlim = (-0.3,0.6))
        else:
            raise Exception(f"Inference mode {self.parameters['inference_mode']} not supported. Supported modes are MAP and MCMC.")

        print("Parameter estimates:")
        print(self.ls.params)
        print("Actual input parameters:")
        print(true_values)

        # Estimation function returns a re-usable LinearSolver object, which we can use to compute the
        # posterior FEM solution conditioned on the data

        # solve for posterior FEM solution conditioned on data

        # plot priors 
        if self.makeplots:
            self.mu, self.Cu = self.ls.solve_prior()
            # mu_f, Cu_f = self.ls.solve_prior_generating()
            self.mu = self._reshape_to_data_obs(self.mu)
            if self.dim == 1:
                plt.figure()
                plt.plot(self.data_coords[:,0],self.mu,'o-',markersize = 2,label = "u")
                plt.fill_between(self.data_coords[:,0],self.mu+1.96*np.diag(self.Cu), self.mu-1.96*np.diag(self.Cu), label = "u 95 confidence",alpha = 0.5)
                # plt.plot(x_data,z_mean, '+-', label = "True z")
                # plt.fill_between(x_data[:,0],z_mean+1.96*np.sqrt(np.diag(z_cov)),z_mean-1.96*np.sqrt(np.diag(z_cov)), label = "True z 95 confidence", alpha = 0.5)
                plt.plot(self.data_coords,self.data_values, '+', label = "data")
                plt.xlabel("x [m]")
                plt.ylabel("y")
                plt.legend()
                plt.title("Prior solutions")

            if self.dim == 2:
                plt.figure()
                plt.tripcolor(self.experiment.mesh.geometry.x[:,0], self.experiment.mesh.geometry.x[:,1], self.problem.u.vector)
                plt.colorbar()
                plt.scatter(self.data_coords[:,0], self.data_coords[:,1], c = self.data_values, cmap="Greys_r")
                plt.colorbar()
                plt.title("Prior FEM solution and data")
            
        self.muy = Function(self.V)

        # solve_posterior computes the full solution on the FEM grid using a Firedrake function
        # the scale_mean option will ensure that the output is scaled to match
        # the data rather than the FEM soltuion

        self.ls.solve_posterior(self.muy, scale_mean=True)
        # self.muy = self._reshape_to_data_obs(self.muy)

        # covariance can only be computed for a select number of locations as covariance is a dense matrix
        # function returns the mean/covariance as numpy arrays, not Firedrake functions

        self.muy2, self.Cuy, self.mu_mesh= self.ls.solve_posterior_covariance(scale_mean=True)
        self.muy2 = self._reshape_to_data_obs(self.muy2)
        self.mu_z2, self.Cu_z2 = self.ls.solve_posterior_real()
        self.mu_z2 = self._reshape_to_data_obs(self.mu_z2)

        # visualize posterior FEM solution and uncertainty

        if self.makeplots:
            if self.dim == 1:
                plt.figure()
                plt.plot(self.data_coords[:,0],self.muy2,'o-',markersize = 0, label = "u posterior scaled")
                plt.fill_between(self.data_coords[:,0],self.muy2+1.96*np.sqrt(np.diag(self.Cuy)), self.muy2-1.96*np.sqrt(np.diag(self.Cuy)), label = "u scaled 95 confidence", alpha = 0.5)
                plt.plot(self.data_coords[:,0],self.mu_z2,'o-',markersize = 2,label = "y from FEM (z+noise) posterior")
                plt.fill_between(self.data_coords[:,0],self.mu_z2+1.96*np.sqrt(np.diag(self.Cu_z2)), self.mu_z2-1.96*np.sqrt(np.diag(self.Cu_z2)), label = "y from FEM (z+noise) 95 confidence", alpha = 0.5)
                plt.plot(self.data_coords,self.data_values, '+', label = "data")
                plt.title("Posterior solutions")
                plt.xlabel("x [m]")
                plt.ylabel("y")
                plt.legend()
                plt.show()

            if self.dim == 2:
                plt.figure()
                plt.tripcolor(self.experiment.mesh.geometry.x[:,0], self.experiment.mesh.geometry.x[:,1],
                            self.muy.x.array)
                plt.colorbar()
                plt.scatter(self.data_coords[:,0], self.data_coords[:,1], c = np.diag(self.Cuy), cmap="Greys_r")
                plt.colorbar()
                plt.title("Posterior FEM solution and uncertainty")
                plt.show()

        # Mark as solved

        self.solved = True

    def solve_lp(self):

        self.problem.A = assemble_matrix(form(self.problem.a), bcs = self.problem.experiment.bcs)
        self.problem.A.assemble()

        self.problem.b = assemble_vector(form(self.problem.L))
        apply_lifting(self.problem.b, [form(self.problem.a)], bcs=[self.problem.experiment.bcs])
        self.problem.b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        set_bc(self.problem.b, self.problem.experiment.bcs)

        # x = Function(self.G.function_space).vector
        x = self.problem.b.copy()
        self.lsolver.setOperators(self.problem.A)
        self.lsolver.solve(self.problem.b, x)
        self.mu = self.ls.im.interp_mesh_to_data(x)


    def _get_default_parameters(self):

        self.parameters = dict()

        # forcing covariance parameters (taken to be known)
        self.parameters['sigma_f'] = [0.3]
        self.parameters['l_f'] = [0.25]

        # true model discrepancy parameters (need to be estimated)
        self.parameters['true_rho'] = np.log(1.0)
        self.parameters['true_sigma_eta'] = np.log(0.0225/2)
        self.parameters['true_l_eta'] = np.log(0.5)

        # data statistical errors (taken to be known)
        self.parameters['sigma_y'] = np.sqrt(2.5e-5)

        # inference parameters
        self.parameters['inference_mode'] = 'MCMC'
        self.parameters['burn in'] = 200
        self.parameters['iterations'] = 2000
        self.parameters['walkers'] = 10
        self.parameters['stabilise'] = True

    def _reshape_to_data_obs(self, array):
        
        return np.reshape(array, np.array(self.data_values).shape)

    def loglikelihood(self, priors = []):

        if not self.solved:
            self.solve()

        self.solve_lp()

        rho, sigma_d, l_d = self.ls.parameters

        if MPI.COMM_WORLD.rank == 0:
            KCu = self.obs_data.calc_K_plus_sigma(self.ls.parameters[1:])
            try:
                L = cho_factor(KCu)
            except LinAlgError:
                raise LinAlgError("Error attempting to factorize the covariance matrix " +
                                  "in model_loglikelihood")
            invKCudata = cho_solve(L, self.obs_data.get_data().reshape((-1,)) - rho*self.mu)
            log_posterior = 0.5*(self.obs_data.get_n_obs()*np.log(2.*np.pi) +
                                 2.*np.sum(np.log(np.diag(L[0]))) +
                                 np.dot(self.obs_data.get_data().reshape((-1,)) - rho*self.mu, invKCudata))
            for i in range(len(priors)):
                if not priors[i] is None:
                    log_posterior -= priors[i].logp(self.params[i])
        else:
            log_posterior = None

        log_posterior = MPI.COMM_WORLD.bcast(log_posterior, root=0)

        assert not log_posterior is None, "error in broadcasting the log likelihood"

        MPI.COMM_WORLD.barrier()

        return log_posterior
