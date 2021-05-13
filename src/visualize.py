import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import tensorflow as tf
from mpl_toolkits.mplot3d import Axes3D     # registers the 3D projection
matplotlib.use('Agg')

def plot(filename):
    # Load point cloud
    pt_cloud = np.load(filename)    # N x 3

    # Separate x, y, z coordinates
    xs = pt_cloud[:, 0]
    ys = pt_cloud[:, 1]
    zs = pt_cloud[:, 2]

    # Plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xs, ys, zs)
    plt.show()

def plot_3d_point_cloud(ax, pc):
    ax.scatter(pc[:,0], pc[:,1], pc[:,2], marker='.')
    ax.view_init(elev=10, azim=240)
    ax.set_xlim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.axis('off')
    return


def reconstruct(original, reconstructed, inds=[], filename='', axis=False):
    original = original.numpy()#.reshape(-1,3)
    reconstructed = np.asarray(reconstructed)#.reshape(-1,3)
    print(original.shape, reconstructed.shape)
    f, axarr = plt.subplots(len(inds), 2, subplot_kw={'projection':'3d'}, figsize=(15,15))
    for i, ind in enumerate(inds):
            axarr[i, 0].scatter(original[ind][:,0], original[ind][:,1], original[ind][:,2], marker='.')
            axarr[i,0].view_init(elev=10, azim=240)
            axarr[i,0].set_xlim([-1, 1])
            axarr[i,0].set_ylim([-1, 1])
            axarr[i,0].set_zlim([-1, 1])
            axarr[i, 1].scatter(reconstructed[ind][:,0], reconstructed[ind][:,1], reconstructed[ind][:,2], marker='.')
            axarr[i,1].view_init(elev=10, azim=240)
            axarr[i,1].set_xlim([-1, 1])
            axarr[i,1].set_ylim([-1, 1])
            axarr[i,1].set_zlim([-1, 1])
            if not axis:
                axarr[i,0].axis('off')
                axarr[i,1].axis('off')

    f.suptitle('Reconstruction')
    axarr[0,0].set_title('Original')
    axarr[0,1].set_title('Reconstruction')
    #f.set_title('Validation Reconstruction')
    #plt.show()
    plt.savefig(filename)
    plt.close()    

def interpolate_from_a_to_b_for_c(encoder, decoder, a=None, b=None, x_c=None, alpha=0.):
    '''Perform interpolation between two classes a and b for any sample x_c.
    model: a trained generative model
    X: data in the original space with shape: (n_samples, n_features)
    labels: array of class labels (n_samples, )
    a, b: class labels a and b
    x_c: input sample to manipulate (1, n_features)
    alpha: scalar for the magnitude and direction of the interpolation
    '''
    # Encode samples to the latent space  
    mean_a, std_a = encoder(a)
    mean_b, std_b = encoder(b)
    noise_a = tf.random.normal(shape=mean_a.shape, mean=0, stddev=1.)
    noise_b = tf.random.normal(shape=mean_b.shape, mean=0, stddev=1.)
    # Find the centroids of the classes a, b in the latent space
    z_a_centoid = tf.reduce_mean(mean_a+tf.exp(std_a*0.5)*noise_a, axis=0)
    z_b_centoid = tf.reduce_mean(mean_b+tf.exp(std_b*0.5)*noise_b, axis=0)
    # The interpolation vector pointing from b -> a
    z_b2a = z_a_centoid - z_b_centoid 
    # Manipulate x_c
    x_c = tf.expand_dims(x_c, axis=0)
    z_mean, z_std = encoder(x_c)
    noise_z = tf.random.normal(shape=z_mean.shape, mean=0, stddev=1.)
    z_c = z_mean + noise_z*tf.exp(0.5*z_std)
    interpolations = []
    val = np.linspace(0,1,alpha)
    for i in range(alpha):
        a = decoder(z_c + val[i] * z_b2a, training=False)[0]
        interpolations.append(a)
    #z_c_interp = tf.expand_dims(z_c_interp, axis=0)
    return decoder(tf.expand_dims(z_a_centoid, axis=0), training=False), decoder(tf.expand_dims(z_b_centoid, axis=0), training=False), interpolations

  
def interpolate_from_a_to_b(encoder, decoder, a=None, b=None, alpha=0., noise=1):
    '''Perform interpolation between two classes a and b for any sample x_c.
    model: a trained generative model
    X: data in the original space with shape: (n_samples, n_features)
    labels: array of class labels (n_samples, )
    a, b: class labels a and b
    x_c: input sample to manipulate (1, n_features)
    alpha: scalar for the magnitude and direction of the interpolation
    '''
    if tf.rank(a)==2 or tf.rank(b)==2:
        a=tf.expand_dims(a, axis=0)
        b=tf.expand_dims(b, axis=0)
    mean_a, std_a = encoder(a)
    mean_b, std_b = encoder(b)
    noise = tf.random.normal(shape=mean_a.shape, mean=0, stddev=1.)
    Z_a = mean_a + tf.exp(std_a*0.5)*noise
    Z_b = mean_b + tf.exp(std_b*0.5)*noise
    z_b2a = Z_a - Z_b
    
    interpolations = []
    val = np.linspace(0,1,alpha)
    for i in range(alpha):
        z_c_interp = Z_a + val[i] * z_b2a
        interpolations.append(decoder(z_c_interp, training=False)[0])
    return interpolations

def interpolate_from_a_to_b_for_c_aae(encoder, decoder, a=None, b=None, x_c=None, alpha=0.):
    '''Perform interpolation between two classes a and b for any sample x_c.
    model: a trained generative model
    X: data in the original space with shape: (n_samples, n_features)
    labels: array of class labels (n_samples, )
    a, b: class labels a and b
    x_c: input sample to manipulate (1, n_features)
    alpha: scalar for the magnitude and direction of the interpolation
    '''
    # Encode samples to the latent space  
    mean_a, std_a = encoder(a)
    mean_b, std_b = encoder(b)
    noise_a = tf.random.normal(shape=mean_a.shape, mean=0, stddev=1.)
    noise_b = tf.random.normal(shape=mean_b.shape, mean=0, stddev=1.)
    # Find the centroids of the classes a, b in the latent space
    z_a_centoid = tf.reduce_mean(mean_a+std_a*noise_a, axis=0)
    z_b_centoid = tf.reduce_mean(mean_b+std_b*noise_b, axis=0)
    # The interpolation vector pointing from b -> a
    #z_b2a = z_a_centoid - z_b_centoid 
    # Manipulate x_c
    x_c = tf.expand_dims(x_c, axis=0)
    z_mean, z_std = encoder(x_c)
    noise_z = tf.random.normal(shape=z_mean.shape, mean=0, stddev=1.)
    z_c = z_mean + noise_z*z_std
    interpolations = []
    val = np.linspace(0,1,alpha)
    for i in range(alpha):
        a = decoder(z_c + z_a_centoid*(1-val[i])+ val[i]*z_b_centoid, training=False)[0]
        interpolations.append(a)
    #z_c_interp = tf.expand_dims(z_c_interp, axis=0)
    return decoder(tf.expand_dims(z_a_centoid, axis=0), training=False), decoder(tf.expand_dims(z_b_centoid, axis=0), training=False), interpolations

  
def interpolate_from_a_to_b_aae(encoder, decoder, a=None, b=None, alpha=0., noise=1):
    '''Perform interpolation between two classes a and b for any sample x_c.
    model: a trained generative model
    X: data in the original space with shape: (n_samples, n_features)
    labels: array of class labels (n_samples, )
    a, b: class labels a and b
    x_c: input sample to manipulate (1, n_features)
    alpha: scalar for the magnitude and direction of the interpolation
    '''
    if tf.rank(a)==2 or tf.rank(b)==2:
        a=tf.expand_dims(a, axis=0)
        b=tf.expand_dims(b, axis=0)
    mean_a, std_a = encoder(a)
    mean_b, std_b = encoder(b)
    noise = tf.random.normal(shape=mean_a.shape, mean=0, stddev=1.)
    z_a = mean_a + std_a*noise
    z_b = mean_b + std_b*noise
    #z_b2a = Z_a - Z_b
    interpolations = []
    val = np.linspace(0,1,alpha)
    for i in range(alpha):
        z_c_interp = z_a*(1-val[i]) + val[i] * z_b
        interpolations.append(decoder(z_c_interp, training=False)[0])
    return interpolations


def interpolate_from_a_to_b_for_c_aae_bin(encoder, decoder, a=None, b=None, x_c=None, alpha=0.):
    '''Perform interpolation between two classes a and b for any sample x_c.
    model: a trained generative model
    X: data in the original space with shape: (n_samples, n_features)
    labels: array of class labels (n_samples, )
    a, b: class labels a and b
    x_c: input sample to manipulate (1, n_features)
    alpha: scalar for the magnitude and direction of the interpolation
    '''
    # Encode samples to the latent space  
    z_a = encoder(a)
    z_b = encoder(b)
    distribution = tf.compat.v1.distributions.Beta(0.01, 0.01)
    noise_a = distribution.sample(z_a.shape)
    noise_b = distribution.sample(z_b.shape)
    #noise_a = tf.random.normal(shape=mean_a.shape, mean=0, stddev=1.)
    #noise_b = tf.random.normal(shape=mean_b.shape, mean=0, stddev=1.)
    # Find the centroids of the classes a, b in the latent space
    z_a_centoid = tf.reduce_mean(z_a, axis=0)
    z_b_centoid = tf.reduce_mean(z_b, axis=0)
    # The interpolation vector pointing from b -> a
    #z_b2a = z_a_centoid - z_b_centoid 
    # Manipulate x_c
    x_c = tf.expand_dims(x_c, axis=0)
    z_c = encoder(x_c)
    #noise_z = tf.random.normal(shape=z_mean.shape, mean=0, stddev=1.)
    #z_c = z_mean + noise_z*z_std
    interpolations = []
    val = np.linspace(0,1,alpha)
    for i in range(alpha):
        a = decoder(z_c + z_a_centoid*(1-val[i])+ val[i]*z_b_centoid, training=False)[0]
        interpolations.append(a)
    #z_c_interp = tf.expand_dims(z_c_interp, axis=0)
    return decoder(tf.expand_dims(z_a_centoid, axis=0), training=False), decoder(tf.expand_dims(z_b_centoid, axis=0), training=False), interpolations

  
def interpolate_from_a_to_b_aae_bin(encoder, decoder, a=None, b=None, alpha=0., noise=1):
    '''Perform interpolation between two classes a and b for any sample x_c.
    model: a trained generative model
    X: data in the original space with shape: (n_samples, n_features)
    labels: array of class labels (n_samples, )
    a, b: class labels a and b
    x_c: input sample to manipulate (1, n_features)
    alpha: scalar for the magnitude and direction of the interpolation
    '''
    if tf.rank(a)==2 or tf.rank(b)==2:
        a=tf.expand_dims(a, axis=0)
        b=tf.expand_dims(b, axis=0)
    z_a = encoder(a)
    z_b = encoder(b)
    #z_b2a = Z_a - Z_b
    interpolations = []
    val = np.linspace(0,1,alpha)
    for i in range(alpha):
        z_c_interp = z_a*(1-val[i]) + val[i] * z_b
        interpolations.append(decoder(z_c_interp, training=False)[0])
    return interpolations
