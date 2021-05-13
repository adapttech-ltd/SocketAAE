import tensorflow as tf
import keras.backend as K
import numpy as np


def anchor_loss(y_true, y_pred, gamma=0.5):
    pred_prob = tf.math.sigmoid(y_pred)

    # Obtain probabilities at indices of true class
    true_mask = tf.dtypes.cast(y_true, dtype=tf.bool)
    q_star = tf.boolean_mask(pred_prob, true_mask)
    q_star = tf.expand_dims(q_star, axis=1)

    # Calculate bce and add anchor loss coeff where labels equal 0
    loss_bce = tf.nn.sigmoid_cross_entropy_with_logits(y_true, y_pred)
    M = 1.0 - y_true
    loss_calc = (M * (1.0 + pred_prob - q_star + 0.05)**gamma + (1.0 - M)) * loss_bce

    return tf.math.reduce_mean(loss_calc)

@tf.function
def pairwise_distances(a, b, p=2):
    """
    Compute the pairwise distance matrix between a and b which both have size [m, n, d] or [n, d]. The result is a tensor of
    size [m, n, n] (or [n, n]) whose entry [m, i, j] contains the distance_tensor between a[m, i, :] and b[m, j, :].
    :param a: A tensor containing m batches of n points of dimension d. i.e. of size [m, n, d]
    :param b: A tensor containing m batches of n points of dimension d. i.e. of size [m, n, d]
    :param p: Norm to use for the distance_tensor
    :return: A tensor containing the pairwise distance_tensor between each pair of inputs in a batch.
    """
    squeezed = False
    if len(a.shape) == 2 and len(b.shape) == 2:
       a = tf.expand_dims(a,0) #[np.newaxis, :, :]
       b = tf.expand_dims(a,0) #b[np.newaxis, :, :]
       squeezed = True
       
    ret = tf.reduce_sum(tf.keras.backend.pow(tf.math.abs(tf.expand_dims(a,2) - tf.expand_dims(b,1)), p), 3)
    #[:, :, np.newaxis, :], [:, np.newaxis, :, :]
    if squeezed:
        ret = tf.squeeze(ret)

    return ret

@tf.function
def chamfer_mean(a,b):
    """
    Compute the chamfer distance between two sets of vectors, a, and b
    :param a: A m-sized minibatch of point sets in R^d. i.e. shape [m, n_a, d]
    :param b: A m-sized minibatch of point sets in R^d. i.e. shape [m, n_b, d]
    :return: A [m] shaped tensor storing the Chamfer distance between each minibatch entry
    """
    #tf.print(a)
    #tf.print(a[0,0,:])
    #tf.print(b[0,0,:])
    M = pairwise_distances(a, b)
    if len(M.shape) == 2:
        M = tf.expand_dims(M,0) #[np.newaxis, :, :]
    #return tf.keras.backend.sum(tf.reduce_sum(tf.reduce_min(M, 1), 1) + tf.reduce_sum(tf.reduce_min(M, 2), 1))
    c=tf.reduce_mean(tf.reduce_min(M, 1), 1) + tf.reduce_mean(tf.reduce_min(M, 2), 1)
    #tf.print(tf.reduce_sum(c))
    return c

@tf.function
def chamfer_sum(a,b):
    """
    Compute the chamfer distance between two sets of vectors, a, and b
    :param a: A m-sized minibatch of point sets in R^d. i.e. shape [m, n_a, d]
    :param b: A m-sized minibatch of point sets in R^d. i.e. shape [m, n_b, d]
    :return: A [m] shaped tensor storing the Chamfer distance between each minibatch entry
    """
    #tf.print(a)
    #tf.print(a[0,0,:])
    #tf.print(b[0,0,:])
    M = pairwise_distances(a, b)
    if len(M.shape) == 2:
        M = tf.expand_dims(M,0) #[np.newaxis, :, :]
    #return tf.keras.backend.sum(tf.reduce_sum(tf.reduce_min(M, 1), 1) + tf.reduce_sum(tf.reduce_min(M, 2), 1))
    c=tf.reduce_sum(tf.reduce_min(M, 1), 1) + tf.reduce_sum(tf.reduce_min(M, 2), 1)
    #tf.print(tf.reduce_sum(c))
    return c


@tf.function
def emd_loss_mean(y_true, y_pred):
    a=tf.reduce_mean(chamfer_mean(y_true, y_pred))
    #tf.print('emd_loss'+ str(a))
    return a

@tf.function
def emd_loss_sum(y_true, y_pred):
    a=tf.reduce_mean(chamfer_sum(y_true, y_pred))
    #tf.print('Recon ')
    #tf.print(a)
    return a

def kl_loss(true, pred):
    # KL divergence loss
    kl_loss = 1 + pred[1] - tf.keras.backend.pow(pred[0],2) - tf.keras.backend.exp(pred[1])
    kl_loss = tf.keras.backend.sum(kl_loss, axis=-1)
    return kl_loss

def autoencoder_loss(inputs, reconstruction):
    return emd_loss_sum(inputs, reconstruction)#mse(inputs, reconstruction)


def discriminator_loss(real_output, fake_output):
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    loss_real = cross_entropy(tf.ones_like(real_output), real_output)
    loss_fake = cross_entropy(tf.zeros_like(fake_output), fake_output)
    return loss_fake + loss_real


def generator_loss(fake_output):
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    return cross_entropy(tf.ones_like(fake_output), fake_output)

