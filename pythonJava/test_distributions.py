import unittest
import numpy as np
import tensorflow_probability as tfp
import action_distributions
import tensorflow as tf
from scipy.stats import norm

class TestDistributions(unittest.TestCase):
    def test_truncatednormal_distribution(self):
        low = 0
        high = 0.009709703
        mussigmoid = 0.46192378
        #mussigmoid = 0.009
        logsigmassigmoid = 0.19137837
        distribution = tfp.distributions.TruncatedNormal(mussigmoid, logsigmassigmoid, low, high)
        sample = 0.00430584
        sample = 0.004
        log_prob = distribution.log_prob(sample)
        print('truncated normal log_prob', log_prob)
        prob = np.exp(log_prob)
        print('truncated normal prob', prob)
        xstep = 0.0001
        x = np.arange(0.0, 0.01+xstep, xstep)
        log_probs = distribution.log_prob(x)
        probs = np.exp(log_probs)
        print('truncated normal probs', probs)
    
    def test_normal_distribution(self):
        mean = 0.51531243
        sigma = 0.2379091
        distribution = tfp.distributions.Normal(mean, sigma)
        xstep = 0.01
        x = np.arange(0.0, 1.0+xstep, xstep)
        log_probs = distribution.log_prob(x)
        probs = np.exp(log_probs)
        print('Normal probs', probs)
        print('Normal probability of mean', distribution.prob(mean))
    
    def test_normal_standard_distribution(self):
        mean = 0
        sigma = 1
        distribution = tfp.distributions.Normal(mean, sigma)
        xstep = 0.01
        x = np.arange(0.0, 1.0+xstep, xstep)
        log_probs = distribution.log_prob(x)
        probs = np.exp(log_probs)
        print('Normal standard probs', probs)
        print('Normal standard probability of', distribution.prob(mean))

    def test_squashed_gaussian4(self):
        low1 = -1
        high1 = 1.0
        means1 = np.array([[0.0]])
        stds1 = np.array([[1.0]])
        squashed_distribution1 = action_distributions.SquashedGaussian(means1, stds1,low=low1,high=high1)
        values1 = np.array([[0.25]])
        outs1 = squashed_distribution1.log_prob(values1)
        print('values', values1,'outs log_prob TEST  squashed gaussian1', outs1, 'prob', tf.exp(outs1))
        xstep = 0.1
        x = np.arange(-1.0-xstep, 1.0+xstep, xstep)
        x = np.reshape(x, (len(x),1))
        log_probs = squashed_distribution1.log_prob(x)
        probs = np.exp(log_probs)
        print('TEST Squashed x', x)
        print('TEST Squashed guassian probs', probs)

    def test_squashed_gaussian3(self):
        low1 = 0
        high1 = 0.5
        means1 = np.array([[0.46192378]])
        stds1 = np.array([[0.19137837]])
        squashed_distribution1 = action_distributions.SquashedGaussian(means1, stds1, low=low1,high=high1)
        values1 = np.array([[0.25]])
        outs1 = squashed_distribution1.log_prob(values1)
        print('outs log_prob  squashed gaussian1', outs1, 'prob', tf.exp(outs1))
        
        low2 = 0
        high2 = 0.2
        means2 = np.array([[0.25]])
        stds2 = np.array([[0.1]])
        squashed_distribution2 = action_distributions.SquashedGaussian(means2, stds2, low=low2, high=high2)
        values2 = np.array([[0.15]])
        outs2 = squashed_distribution2.log_prob(values2)
        print('outs log_prob  squashed gaussian2', outs2, 'prob', tf.exp(outs2))

        low12 = np.array([[0], [0]]) 
        high12 = np.array([[0.5], [0.2]]) 
        means12 = np.array([[0.46192378], [0.25]])
        stds12 = np.array([[0.19137837], [0.1]])
        squashed_distribution12 = action_distributions.SquashedGaussian(means12, stds12,low=low12,high=high12)
        values12 = np.array([[0.25],[0.15]])
        outs12 = squashed_distribution12.log_prob(values12)
        print('outs log_prob  squashed gaussian12', outs12, 'prob', tf.exp(outs12))

        lowjoint = np.array([[0, 0]])
        highjoint = np.array([[0.5, 0.2]])
        meansjoint = np.array([[0.46192378, 0.25]])
        stdsjoint = np.array([[0.19137837, 0.1]])
        squashed_distributionjoint = action_distributions.SquashedGaussian(meansjoint, stdsjoint,low=lowjoint,high=highjoint)
        valuesjoint = np.array([[0.25, 0.15]])
        outsjoint = squashed_distributionjoint.log_prob(valuesjoint)
        print('outs log_prob  squashed gaussianjoint', outsjoint, 'prob', tf.exp(outsjoint))

    def test_squashed_gaussian2(self):
        low = 0
        high = 0.009709703
        means = np.array([[0.46192378]])
        stds = np.array([[0.19137837]])
        stds = np.exp(stds)
        squashed_distribution = action_distributions.SquashedGaussian(means,stds, low=low,high=high)
        
        # Values to get log-likelihoods for.
        values = np.array([[0.004]])
        # Unsquash values, then get log-llh from regular gaussian.
        unsquashed_values = np.arctanh((values - low) / (high - low) * 2.0 - 1.0)
        log_prob_unsquashed = np.sum(np.log(norm.pdf(unsquashed_values, means, stds)), -1)
        log_prob = log_prob_unsquashed - np.sum(np.log(1 - np.tanh(unsquashed_values) ** 2), axis=-1)
        print('values', values)
        outs = squashed_distribution.log_prob(values)
        print('outs log_prob  squashed gaussian', outs, 'prob', tf.exp(outs))
        print('log_prob squashed gaussian', log_prob)
        xstep = 0.0001
        x = np.reshape(np.arange(0.0, 0.01+xstep, xstep), (101,1))
        log_probs = squashed_distribution.log_prob(x)
        probs = np.exp(log_probs)
        print('Squashed x', x)
        print('Squashed guassian probs', probs)
        xstep = 0.1
        x = np.reshape(np.arange(0.0, 1.0+xstep, xstep), (11,1))
        log_probs = squashed_distribution.log_prob(x)
        probs = np.exp(log_probs)
        print('Squashed x', x)
        print('Squashed guassian probs', probs)

   

    def test_normal_distribution_hm(self):
# NN output.
        means = np.array([[0.1, 0.2, 0.3, 0.4, 50.0], [-0.1, -0.2, -0.3, -0.4, -1.0]], dtype=np.float32)
        log_stds = np.array([[0.8, -0.2, 0.3, -1.0, 2.0], [0.7, -0.3, 0.4, -0.9, 2.0]],dtype=np.float32)
        inputs=np.concatenate([means, log_stds], axis=-1)
        print('inputs', inputs)
        mean, log_std = tf.split(inputs, 2, axis=1)
        # Convert to parameters for distr.
        stds = np.exp(log_stds)
        diag_distribution = action_distributions.DiagGaussian(mean, log_std)
        # Convert to parameters for distr
        # Values to get log-likelihoods for.
        values = np.array( [[0.9, 0.2, 0.4, -0.1, -1.05], [-0.9, -0.2, 0.4, -0.1, -1.05]])
        # get log-llh from regular gaussian.
        log_prob = np.sum(np.log(norm.pdf(values, means, stds)), -1)
        outs = diag_distribution.log_prob(values)
        print('DiagGaussian logp', outs)
        print('Correct DiagGaussian logp', outs)    
    #def test_normal_distribution_hm(self):
        #mean = np.array([0.51531243, 0.51531243])
        #sigma = np.array([ 0.2379091, 0.2379091])
        #distribution = action_distributions.DiagGaussian(mean, sigma)
        #xstep = 0.01
        #x = np.arange(0.0, 1.0+xstep, xstep)
        #log_probs = distribution.log_prob(x)
        #probs = np.exp(log_probs)
        #print('Normal HM probs', probs)
        #print('Mean probability', np.exp(distribution.log_prob(mean)))

if __name__ == '__main__':
    unittest.main()

