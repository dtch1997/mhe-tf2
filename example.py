# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 16:54:12 2020

@author: Daniel Tan
"""


import tensorflow as tf
from hyperspherical_energy import HypersphericalEnergy

layer = tf.keras.layers.Dense(50, kernel_regularizer=HypersphericalEnergy())
tensor = tf.ones(shape=(10,10))
out = layer(tensor)

# Verify that hyperspherical loss has been registered
print(layer.losses) 

