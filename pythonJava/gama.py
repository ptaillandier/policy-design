import tensorflow as tf
from keras import Sequential


### Define the gama agent network  ###
# This network will take as input an observation of the environment and output the probability of taking each of the possible actions. 
def create_model(n_observations, n_actions) -> Sequential:
    # The Sequential model definition takes a list of layers as an argument, specifying the calculation order from the input to the output.
    model = tf.keras.models.Sequential([
        # First Dense layer
        tf.keras.layers.Dense(units=32, activation='relu', input_shape=(n_observations,)),
        # Second Dense layer
        tf.keras.layers.Dense(units=32, activation='relu'),
        # The last Dense layer, which will provide the network's output (e.g. the output action probabilities)  it is set to the number of actions multiplied by two since we want to use a probability distribution (mean, variance) for each continuous action
        tf.keras.layers.Dense(units=n_actions*2, activation=None)
    ])
    return model
