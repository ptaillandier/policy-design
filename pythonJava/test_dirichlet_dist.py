import unittest
import numpy as np
import tensorflow_probability as tfp
import action_distributions
import tensorflow as tf
from scipy.stats import norm

class TestDirichlet(unittest.TestCase):

     def test_dirichlet_distribution(self):
        # NN output.
        log_concentrations = np.array([[0.8, -0.2, 0.3, -1.0], [0.7, -0.3, 0.4, 2.0]],dtype=np.float32)
        # Convert to parameters for distr.
        concentrations = np.exp(log_concentrations)
        print('concentrations', concentrations)
        distribution = action_distributions.Dirichlet(log_concentrations)
        values = distribution.sample()
        print('sampled values', values)
        # Convert to parameters for distr
        # Values to get log-likelihoods for.
        #values = np.array( [[0.9, 0.2, 0.4, -0.1, -1.05], [-0.9, -0.2, 0.4, -0.1, -1.05]])
        # get log-llh from regular gaussian.
        outs = distribution.log_prob(values)
        print('Dirichlet for values logp', outs, ' probs', distribution.prob(values))
        #print('Correct DiagGaussian logp', outs)

if __name__ == '__main__':
    unittest.main()

    
