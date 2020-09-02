# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 15:50:10 2020

@author: Daniel Tan
"""


import tensorflow as tf
from math import pi as PI

def _compute_norm(x, axis=[0], keepdims=False, epsilon=1e-12):
    """
    Compute the norm of x along the specified axis / axes. 
    """
    x_norm_squared = tf.math.reduce_sum(x*x, axis=axis, keepdims=keepdims)
    return tf.math.sqrt(x_norm_squared + epsilon)
        

@tf.keras.utils.register_keras_serializable(package='Custom', name='hyperspherical_energy')
def hyperspherical_energy(weight_matrix, axis=0, power=0, 
                          use_half_mhe=False, use_arccos=False, 
                          epsilon = 1e-4):
    """
    Parameters:
        weight_matrix: Tensor of layer weights. 
            One dimension should correspond to "number of neurons" and the remaining are arbitrary.
            
        power: 
            The power in the nonlinearity function. 
        
        axis:
            Dimension which corresponds to "number of neurons". 
            
        use_half_mhe: 
            Set True for half-MHE loss. Default: False. 
        
        use_arccos:
            Whether to use arccos in the nonlinearity function
            
        epsilon:
            Small value to add in tf.math.sqrt and tf.math.acos to avoid numerical instability
        
    Return:
        l, Hyperspherical energy as described in https://arxiv.org/pdf/1805.09298.pdf
        l is a tensor with shape (). I.e. scalar
    """
    
    # Some basic error checking
    if power < 0: 
        raise ValueError(f"Parameter power={power}; expected power >= 0")
    if axis >= len(weight_matrix.shape):
        raise ValueError(f"Parameter axis={axis} is invalid for weight_matrix of shape {weight_matrix.shape}")    
    
    # Compute the normalized inner product of each pair of columns
    num_filters = weight_matrix.shape[axis]
    # Each weight is one column in the resulting matrix
    weight_matrix = tf.reshape(weight_matrix, [-1, num_filters])
    if use_half_mhe:
        # Concatenate negative weights and apply full MHE to the concatenation
        weight_matrix_neg = weight_matrix * -1
        weight_matrix = tf.concat((weight_matrix, weight_matrix_neg), axis=1)
        num_filters *= 2
    
    weight_norms = _compute_norm(weight_matrix, axis=[0], keepdims=True, epsilon=epsilon)
    weight_norm_pairwise = tf.linalg.matmul(tf.transpose(weight_norms), weight_norms)
    inner_product_pairwise = tf.linalg.matmul(tf.transpose(weight_matrix), weight_matrix)
    inner_product_normalized = inner_product_pairwise / weight_norm_pairwise
    
    cross_terms = None
    if use_arccos:
        cross_terms = tf.math.acos(inner_product_normalized) / PI + epsilon
    else:
        cross_terms = 2.0 - 2.0 * inner_product_normalized + tf.linalg.diag([1.0] * num_filters)
        # For non-arccos, the power is halved.
        power = power / 2
    
    final = cross_terms
    if power != 0:
        final = tf.pow(final, tf.ones_like(cross_terms) * (-power))
    final -= tf.linalg.band_part(final, -1, 0)
    count = num_filters * (num_filters - 1) / 2.0
    
    loss =  1 * tf.math.reduce_sum(final) / count
    return loss
    


@tf.keras.utils.register_keras_serializable(package='Custom', name='HypersphericalEnergy')
class HypersphericalEnergy(tf.keras.regularizers.Regularizer):
    """
    Hyperspherical energy as described in https://arxiv.org/pdf/1805.09298.pdf
    
    The hyperspherical energy is conceived as a loss on weights
    To use, set 'kernel_regularizer=HypersphericalEnergy()' in tf.keras.layer construction
    """
    def __init__(self, axis=0, power=0, 
                 use_half_mhe=False, use_arccos=False, 
                 epsilon = 1e-4):
        self.axis = axis
        self.power = power
        self.use_half_mhe = use_half_mhe
        self.use_arccos= use_arccos
        self.epsilon = epsilon
    
    def __call__(self, x):
        return hyperspherical_energy(x, 
                                     axis = self.axis,
                                     power = self.power,
                                     use_half_mhe = self.use_half_mhe,
                                     use_arccos = self.use_arccos,
                                     epsilon = self.epsilon)
    
    def get_config(self):   
        return {
            'axis': self.axis,
            'power': self.power,
            'use_half_mhe': self.use_half_mhe,
            'use_arccos': self.use_arccos,
            'epsilon': self.epsilon
        }
    
    @classmethod
    def from_config(config):
        raise NotImplementedError()

    
