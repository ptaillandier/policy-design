import tensorflow as tf
from keras import Sequential
from numpy.random import seed


### Define the gama agent network  ###
# This network will take as input an observation of the environment and output the probability of taking each of the possible actions. 
def create_model(n_observations, n_actions) -> Sequential:
    seed(1)
    tf.random.set_seed(1)
    # The Sequential model definition takes a list of layers as an argument, specifying the calculation order from the input to the output.
    model = tf.keras.models.Sequential([
        # First Dense layer
        tf.keras.layers.Dense(units=32, activation='relu', input_shape=(n_observations,)),
        # The last Dense layer, which will provide the network's output (e.g. the output action probabilities)  it is set to the number of actions (2)
        tf.keras.layers.Dense(units=n_actions, activation=None)
        # Note: using linear output (no activation function) because using softmax activation and SparseCategoricalCrossentropy() has issues and which are patched by the tf.keras model.
        # A safer approach, in general, is to use a linear output (no activation function) with SparseCategoricalCrossentropy(from_logits=True).
    ])
    return model
