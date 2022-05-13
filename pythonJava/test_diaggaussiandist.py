import unittest
import numpy as np
import tensorflow_probability as tfp
import action_distributions
import tensorflow as tf
from scipy.stats import norm

class TestDiagGaussian(unittest.TestCase):

     def test_normal_distribution(self):
        mean = 0.0
        sigma = 1.0
        distribution = tfp.distributions.Normal(mean, sigma)
        xstep = 0.01
        x = np.arange(0.0, 1.0+xstep, xstep)
        log_probs = distribution.log_prob(x)
        probs = np.exp(log_probs)
        print('Normal with mean', mean, 'std', sigma,' probs', probs)
        print('Normal probability of mean', distribution.prob(mean))
        means = np.array([[mean]], dtype=np.float32)
        stds = np.array([[sigma]], dtype=np.float32)
        diag_distribution = action_distributions.DiagGaussian(means, stds)
        x = np.reshape(x, (len(x),1))
        log_probs = diag_distribution.log_prob(x)
        probs = np.exp(log_probs)
        print('Diag dist with mean', mean, 'std', sigma,' probs', probs)
        print('Diag dist with mean', mean, 'std', sigma, 'log_probs', log_probs)
        print('Diag dist probability of mean', distribution.prob(mean))

     def test_diaggaussian_distribution(self):
        # NN output.
        means = np.array([[0.1, 0.2, 0.3, 0.4, 50.0], [-0.1, -0.2, -0.3, -0.4, -1.0]], dtype=np.float32)
        log_stds = np.array([[0.8, -0.2, 0.3, -1.0, 2.0], [0.7, -0.3, 0.4, -0.9, 2.0]],dtype=np.float32)
        # Convert to parameters for distr.
        stds = np.exp(log_stds)
        print('stds', stds)
        diag_distribution = action_distributions.DiagGaussian(means, stds)
        # Convert to parameters for distr
        # Values to get log-likelihoods for.
        values = np.array( [[0.9, 0.2, 0.4, -0.1, -1.05], [-0.9, -0.2, 0.4, -0.1, -1.05]])
        # get log-llh from regular gaussian.
        log_prob = np.sum(np.log(norm.pdf(values, means, stds)), -1)
        outs = diag_distribution.log_prob(values)
        print('DiagGaussian logp', outs, ' probs', diag_distribution.prob(values))
        print('Correct DiagGaussian logp', outs)

if __name__ == '__main__':
    unittest.main()

    
