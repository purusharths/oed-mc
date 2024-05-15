import qmcpy as qm

# Define parameters for the normal distribution (mean, standard deviation)
mu = 5.0
sigma = 1.0

# Sample size (number of samples to draw)
num_samples = 1000

# Create a Normal distribution object
normal_dist = qm.true_measure.gaussian.Normal(sampler="TrueMeasure")

# Draw samples from the normal distribution
samples = qm.sample(normal_dist, size=num_samples)

# Print the first 5 samples (optional)
print(f"First 5 samples: {samples[:5]}")

# Further analysis (e.g., calculate statistics, plot the distribution)
# ...
