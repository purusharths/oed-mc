import umbridge
import argparse
import qmcpy as qp
from qmcpy.integrand.um_bridge_wrapper import UMBridgeWrapper
import numpy as np
import umbridge
import random 
# Set up umbridge model and (optional) model config
# model = umbridge.HTTPModel("http://localhost:4242", "forward")
config = {}

class TestModel(umbridge.Model):
    def __init__(self):
        super().__init__("forward") # Give a name to the model
        #self.model = umbridge.Model

    def get_input_sizes(self, config):
        return [2]

    def get_output_sizes(self, config):
        return [1]

    def __call__(self, parameters, config):
        print(parameters)
        # # Get input dimension from model
        d = self.get_input_sizes(config)[0]
        print("input size: {}".format(d))
        # Choose a distribution of suitable dimension to sample via QMC
        dnb2 = qp.DigitalNetB2(d)
        print("dnb2")
        gauss_sobol = qp.Uniform(dnb2, lower_bound=[1]*d, upper_bound=[1.05]*d)
        print("gauss_sobol")
        # print(gauss_sobol)
        # # Create integrand based on umbridge model
        integrand = UMBridgeWrapper(true_measure=gauss_sobol, config=config, parallel=False, model=self)
        print("wrapper")
        # # Run QMC integration to some accuracy and print results
        qmc_sobol_algorithm = qp.CubQMCSobolG(integrand, abs_tol=1e-1)
        print("algo")
        solution,data = qmc_sobol_algorithm.integrate()
        print(data)
        print("completed")
        output = parameters[0][0] * 2 # Simply multiply the first input entry by two.
        return [[output]]

    def supports_evaluate(self):
        return True

    def gradient(self, out_wrt, in_wrt, parameters, sens, config):
        return [2*sens[0]]

    def supports_gradient(self):
        return True
    
    # def __call__(self, parameters, config):
    #     samples = np.random.normal(parameters[0][0], 5, size=10)
    #     print(random.choice(samples))
    


# Get input dimension from model
# d = model.get_input_sizes(config)[0]

# # Choose a distribution of suitable dimension to sample via QMC
# dnb2 = qp.DigitalNetB2(d)
# gauss_sobol = qp.Uniform(dnb2, lower_bound=[1]*d, upper_bound=[1.05]*d)

# # Create integrand based on umbridge model
# integrand = UMBridgeWrapper(gauss_sobol, model, config, parallel=False)

# # Run QMC integration to some accuracy and print results
# qmc_sobol_algorithm = qp.CubQMCSobolG(integrand, abs_tol=1e-1)
# solution,data = qmc_sobol_algorithm.integrate()
# print(data)
    
testmodel = TestModel()
a = umbridge.serve_models([testmodel], 4442)
print(a)