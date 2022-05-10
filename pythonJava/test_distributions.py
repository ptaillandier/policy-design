import unittest
import numpy as np
import tensorflow_probability as tfp

class TestDistributions(unittest.TestCase):
    def test_distribution(self):
        low = 0
        high = 0.009709703
        mussigmoid = 0.46192378
        #mussigmoid = 0.009
        logsigmassigmoid = 0.19137837
        distribution = tfp.distributions.TruncatedNormal(mussigmoid, logsigmassigmoid, low, high)
        sample = 0.00430584
        sample = 0.004
        log_prob = distribution.log_prob(sample)
        print('log_prob', log_prob)
        prob = np.exp(log_prob)
        print('prob', prob)
        prob =  distribution.prob(sample)
        print('prob2', prob)
        xstep = 0.0001
        x = np.arange(0.0, 0.01+xstep, xstep)
        log_probs = distribution.log_prob(x)
        probs = np.exp(log_probs)
        print('probs', probs)
        probs = distribution.prob(x)
        print('probs2', probs)

if __name__ == '__main__':
    unittest.main()

