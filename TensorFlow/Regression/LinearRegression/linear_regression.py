# A simple linear regression model.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

# Input house area parameters and predict prices.
area = tf.constant([[50], [100], [150], [200]], dtype = tf.float32)
price_true = tf.constant([[120], [200], [280], [360]], dtype = tf.float32)

# Linear model with 1 layer, 1 input parameter (area) and 1 output (price).
linear_model = tf.layers.Dense(units = 1, use_bias = True, name = "linear_model")
price_prediction = linear_model(area)

# Use mean squared error as the loss function and RMSProp as the optimizer (learning rate is 0.005).
loss = tf.losses.mean_squared_error(labels=price_true, predictions=price_prediction)
optimizer = tf.train.RMSPropOptimizer(0.005)
train = optimizer.minimize(loss)

# Initialize the layer.
variable_initialization = tf.global_variables_initializer()
model_session = tf.Session()
model_session.run(variable_initialization)

# Train for 10000 times.
for i in range(10000):
    _, loss_value = model_session.run((train, loss))

# Print the kernel and bias.
print("The kernel of the model is: ")
print(model_session.run(linear_model.kernel))
print("The bias of the model is: ")
print(model_session.run(linear_model.bias))

# Print the predicted prices.
print("If the area is 250, the predicted price is: ")
print(model_session.run(linear_model(tf.constant([[250]], dtype = tf.float32))))

# Save the computation graph to a TensorBoard summary file.
#writer = tf.summary.FileWriter('.')
#writer.add_graph(tf.get_default_graph())