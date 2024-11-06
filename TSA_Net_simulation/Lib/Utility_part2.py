import tensorflow as tf
import numpy as np
from Lib.ms_ssim import *
    
def loss_mse(decoded, ground):
    loss_pixel = tf.square(tf.subtract(decoded, ground))
    return tf.reduce_mean(loss_pixel)
    
def loss_rmse(decoded, ground):
    loss_pixel = tf.square(tf.subtract(decoded, ground))
    loss_pixel = tf.sqrt(tf.reduce_mean(loss_pixel,axis=(1,2,3)))
    return tf.reduce_mean(loss_pixel)

def loss_spec(decoded, ground):
    grad_ground = tf.subtract(ground[:,:,:,1:],ground[:,:,:,:-1])
    grad_decode = tf.subtract(decoded[:,:,:,1:],decoded[:,:,:,:-1])
    loss_pixel = tf.reduce_mean(tf.square(tf.subtract(grad_ground, grad_decode)),axis=(1,2,3))
    return tf.reduce_mean(tf.sqrt(loss_pixel)) #tf.reduce_mean(loss_pixel)

def loss_SSIM(decoded,ground):
    return MultiScaleSSIM(decoded,ground)    

def loss_multi_scale_structure(y_pred, y_true):
    """
    Implements the multi-scale structure loss function.
    
    Args:
        y_pred (tf.Tensor): The predicted output tensor.
        y_true (tf.Tensor): The true output tensor.
        
    Returns:
        tf.Tensor: The multi-scale structure loss.
    """
    loss = 0
    for i in range(5):
        # Downsample the predicted and true outputs
        y_pred_down = tf.image.resize(y_pred, (y_pred.shape[1] // (2 ** i), y_pred.shape[2] // (2 ** i)))
        y_true_down = tf.image.resize(y_true, (y_true.shape[1] // (2 ** i), y_true.shape[2] // (2 ** i)))
        
        # Compute the SSIM between the downsampled outputs
        ssim = tf.image.ssim(y_pred_down, y_true_down, max_val=1.0)
        
        # Add the weighted SSIM to the loss
        loss += (0.5 ** i) * (1 - ssim)
    
    return loss

def loss_content_aware_spectral(y_pred, y_true):
    """
    Implements the content-aware spectral loss function.
    
    Args:
        y_pred (tf.Tensor): The predicted output tensor.
        y_true (tf.Tensor): The true output tensor.
        
    Returns:
        tf.Tensor: The content-aware spectral loss.
    """
    # Convert the input tensors to complex64 data type
    y_pred_complex = tf.complex(y_pred, tf.zeros_like(y_pred))
    y_true_complex = tf.complex(y_true, tf.zeros_like(y_true))
    
    # Compute the FFT of the predicted and true outputs
    fft_pred = tf.abs(tf.signal.fft2d(y_pred_complex))
    fft_true = tf.abs(tf.signal.fft2d(y_true_complex))
    
    # Compute the variance of the true FFT components
    fft_true_var = tf.math.reduce_variance(fft_true, axis=[1, 2], keepdims=True)
    
    # Compute the weights for each frequency component
    eps = 1e-8
    weights = fft_true_var / (eps + fft_true_var)
    
    # Compute the content-aware spectral loss
    loss = tf.math.reduce_mean(weights * tf.math.square(fft_pred - fft_true))
    
    return loss

def tensor_log10(x):
    numerator = tf.log(x)
    denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
    return  tf.divide(numerator,denominator)

def metric_psnr(decoded, ground):
    loss_pixel = tf.reduce_mean(tf.square(tf.subtract(decoded, ground)),axis=(1,2,3))
    psnr_s = tf.constant(10.0)*tensor_log10(tf.square(tf.reduce_max(ground,axis=(1,2,3)))/loss_pixel)
    return tf.reduce_mean(psnr_s)

def metric_ssim(decoded, ground):
    loss_pixel = tf.abs(tf.subtract(decoded, ground))
    return tf.reduce_mean(loss_pixel)

def calculate_metrics(decoded, ground):
    psnr = metric_psnr(decoded, ground)
    ssim = metric_ssim(decoded, ground)
    return psnr, ssim
