from umbridge.pymc import UmbridgeOp
import numpy as np
import pymc as pm
from pytensor import tensor as pt
import arviz as az
import matplotlib.pyplot as plt

# Connect to model specifying model's URL and name
op = UmbridgeOp("http://127.0.0.1:4232", "forward")

# print(dir(op))

# print(op.get_input_sizes())
# print(op.get_output_sizes())
input_dim = 2
input_val = [100.0, 10.0]

op_application = op(pt.as_tensor_variable(input_val))
print(f"Model output: {op_application.eval()}")


with pm.Model() as model:
    posterior = pm.DensityDist('posterior',logp=op,shape=input_dim)
    map_estimate = pm.find_MAP()
    print(f"MAP estimate of posterior is {map_estimate['posterior']}")
    inferencedata = pm.sample(draws=50)
    az.plot_pair(inferencedata)
    plt.show()