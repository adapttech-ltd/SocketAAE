import tensorflow as tf
import numpy as np
import os
from pathlib import Path
import matplotlib.patches as mpatches
from sklearn import manifold
import matplotlib.pyplot as plt
import sys
import argparse

from src.dataset_utils import split_files, test_dataloader
from src.visualize import reconstruct, interpolate_from_a_to_b_aae, interpolate_from_a_to_b_for_c_aae, plot_3d_point_cloud
from src.model import make_encoder_model, make_decoder_model, make_discriminator_model
from src.losses import autoencoder_loss, discriminator_loss, generator_loss
from src.evaluation_metrics import cm_mmd, cm_cov, jsd_between_point_cloud_sets

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--model_path', '-mp', type=str, required=True,
                    help='path to model folder')
parser.add_argument('--dataset_path', '-dp', type=str, required=True,
                    help='path to dataset folder')
parser.add_argument('--figure_path', '-fp', type=str, default=os.getcwd()+'/figures',
                    help='path to figures')
parser.add_argument('--input_shape', '-in_shape', type=int, default=2048,
                    help='number of point in each point cloud')
parser.add_argument('--z_dim', '-zd', type=int, default=256,
                    help='autoencoder bottleneck dimension')
parser.add_argument('--normalization', '-norm', type=int, default=0,
                    help='normalize each point cloud to the unit sphere')
args = parser.parse_args()


if not os.path.exists(args.figure_path):
    os.makedirs(args.figure_path)

#DEFINE MODELS
encoder = make_encoder_model(args.input_shape, args.z_dim)
decoder = make_decoder_model(args.input_shape, args.z_dim)
discriminator = make_discriminator_model(args.z_dim)

encoder.load_weights(args.model_path+'/encoder')
decoder.load_weights(args.model_path+'/decoder')
discriminator.load_weights(args.model_path+'/discriminator')

test_files = []
train_files = []
filepath = args.dataset_path+'/test_files.txt'
if not os.path.exists(filepath):
    print('Test dataset split does not exist in the indicated path.')
    sys.exit(1)

with open(filepath) as fp:
   line = fp.readline()
   while line:
        line = fp.readline()
        test_files.append(line.split('\n')[0])
print('Test Files: ', len(test_files))

print('Loading dataset')
test_ds = test_dataloader(test_files, normalization=args.normalization)

print('Loading test data')
x_test = tf.concat([y for y,_ in test_ds], axis=0)
print(x_test.shape)

print('Loading test labels')
test_y = tf.concat([k for _,k in test_ds], axis=0)
test_y = test_y.numpy()


batch = 5
z_mean = []
z_std = []
code = []
x_test_decoded = []
x_train_decoded = []
digits = [i for i in range(min(batch, len(x_test)))] # how many digits we will display
print('Decoding test')
for i in range(0,len(x_test)):
    mean, std = encoder(x_test[i:i+1], training=False)
    noise = tf.random.normal(shape=mean.shape, mean=0, stddev=1.)
    z_mean.extend(mean)
    z_std.extend(std)
    code.extend(mean + std*noise)
    x_test_decoded.extend(decoder(mean + std*noise, training=False))

print('Reconstruction metrics!')
#Metrics Reconstruction
mmd_test = cm_mmd(x_test, np.asarray(x_test_decoded),x_test.shape[0], batch)
print('MMD test reconstruction ', mmd_test)

print('Generative metrics!')
#Metrics Generative
mmd_test = []
cov_test = []
for i in range(3):
    print(i)
    fixed_noise_sampling = tf.random.normal(shape=(3*len(x_test),args.z_dim), mean=0, stddev=0.2)
    x_sampled = decoder(fixed_noise_sampling, training=False).numpy().squeeze() 
    mmd = cm_mmd(x_test, x_sampled,x_test.shape[0], batch)
    print('MMD generative', mmd)
    cov = cm_cov(x_test, x_sampled, x_test.shape[0], batch)
    print('COV generative', cov)
print('Testing finished! Saving images')

# Latent interpolation Unique
start = x_test[0]
end = x_test[1]
interpolations = interpolate_from_a_to_b_aae(encoder, decoder,a=start, b=end, alpha=5)
f, axarr = plt.subplots(5,2, subplot_kw={'projection':'3d'}, figsize=(15,15))
for i in range(5):
    plot_3d_point_cloud(axarr[i,0], interpolations[i])
plot_3d_point_cloud(axarr[0,1], start)
plot_3d_point_cloud(axarr[4,1], end)

f.suptitle('Interpolation between two instances')
axarr[0,0].set_title('Interpolation')
axarr[0,1].set_title('Starting Point')
axarr[4,1].set_title('Final Point')
plt.savefig(args.figure_path+'/interpolation_unique.png')
plt.close('all')


# Reconstruction
reconstruct(x_test, x_test_decoded, digits, filename=args.figure_path+'/reconstruction.png')#str(epoch))

# Sampling
fig, ax = plt.subplots(5, 5, subplot_kw={'projection':'3d'}, figsize=(15,15))
f.suptitle('Sampling')
fixed_noise_sampling = tf.random.normal(shape=(25,args.z_dim), mean=0, stddev=0.2)
x_sampled = decoder(fixed_noise_sampling, training=False).numpy().squeeze()         
cnt = 0
for j in range(5):
    for i in range(5):
        ax[j,i].scatter(x_sampled[cnt][:,0], x_sampled[cnt][:,1], x_sampled[cnt][:,2], marker='.')
        ax[j,i].axis('off')
        cnt=cnt+1
plt.savefig(args.figure_path+'/sampling.png')
plt.close('all')
