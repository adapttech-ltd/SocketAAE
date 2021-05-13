import argparse
from sklearn.decomposition import PCA
import numpy as np
import tensorflow as tf
import datetime
import os
import sys
import math
from sklearn import manifold
import matplotlib.pyplot as plt
import matplotlib
import time
from glob import glob
from pathlib import Path
import matplotlib.patches as mpatches
from sklearn.model_selection import StratifiedKFold
from tensorflow import keras

from src.dataset_utils import split_files, dataloader_w_classes, dataloader
from src.visualize import reconstruct, interpolate_from_a_to_b_aae, interpolate_from_a_to_b_for_c_aae, plot_3d_point_cloud
from src.model import make_encoder_model, make_decoder_model, make_discriminator_model
from src.losses import autoencoder_loss, discriminator_loss, generator_loss
from src.evaluation_metrics import cm_mmd, cm_cov, jsd_between_point_cloud_sets

parser = argparse.ArgumentParser(description='')
parser.add_argument('--dataset_path', '-dp', type=str, required=True,
                    help='path to dataset folder')
parser.add_argument('--model_name', '-n', type=str, required=True,
                    help='model identifier')
parser.add_argument('--termination', '-term', type=str, default='npy',
                    help='data file termination to search for')
parser.add_argument('--augmentation', '-aug', type=int, default=0,
                    help='perform on the fly augmentation at each epoch')
parser.add_argument('--normalization', '-norm', type=int, default=0,
                    help='normalize each point cloud to the unit sphere')
parser.add_argument('--input_shape', '-in_shape', type=int, default=2048,
                    help='number of point in each point cloud')
parser.add_argument('--batch_size', '-bs', type=int, default=1,
                    help='train batch size')
parser.add_argument('--val_batch_size', '-v_bs', type=int, default=1,
                    help='val batch size')
parser.add_argument('--epochs', '-ep', type=int, default=1000,
                    help='number of epochs to train the model')
parser.add_argument('--data_split', '-d_split', type=int, nargs='+', default=[0.8,0.1,0.1],
                    help='train val test split')
parser.add_argument('--z_dim', '-zd', type=int, default=256,
                    help='autoencoder bottleneck dimension')
parser.add_argument('--load_weights_from', '-tfl', type=str, default='None',
                    help='load weights from path')
parser.add_argument('--freeze_layers', '-fl', type=int, nargs='+', default=[0,0,0],
                    help='freeze encoder, decoder and discriminator layers until layer N')
parser.add_argument('--learning_rate', '-lr', type=float, default=1e-4,
                    help='base learning rate')
parser.add_argument('--learning_rate_factors', '-lr_f', type=float, nargs='+', default=[2,0.25,0.5],
                    help='learning rate factors for autoencoder, discriminator and generator respectively')
parser.add_argument('--adam_beta1', '-b1', type=float, default=0.5,
                    help='Adam optimizer beta 1 value')
parser.add_argument('--adam_beta2', '-b2', type=float, default=0.999,
                    help='Adam optimizer beta 2 value')
parser.add_argument('--weights_losses', '-ww', type=float, nargs='+', default=[0.05,1.,1.],
                    help='Weights for autoencoder, discriminator and generator losses respectively')
parser.add_argument('--save_frequency', '-sf', type=float, default=15,
                    help='Save frequency for model')
args = parser.parse_args()
#---------------------------------------------------------------------------------------------------------------------------
#Learning Rate Decay ## Did not move into argparse yet because maybe it should just be deleted? not sure if working
DECAY = False
reg_start = 150
decay_rate = 5000000
growth_rate = 0.02

#----------------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------
#LOADING DATA

print('Creating train and val datasets...')
filepath = args.dataset_path+'/train_files.txt'
if os.path.exists(filepath):
    train_files = []
    val_files = []
    
    with open(filepath) as fp:
        line = fp.readline()
        while line:
            line = fp.readline()
            train_files.append(line.split('\n')[0])

    val_filepath = args.dataset_path+'/val_files.txt'
    with open(val_filepath) as fp:
        line = fp.readline()
        while line:
            line = fp.readline()
            val_files.append(line.split('\n')[0])
            
else:
    train_files, val_files, test_files = split_files(args.dataset_path, args.termination, args.data_split[0], args.data_split[1], args.data_split[2])
    for filename, examples in zip(["train_files.txt","val_files.txt", "test_files.txt"],
                                    [train_files, val_files, test_files]):
        file_obj = open(os.path.join(os.getcwd(), args.dataset_path+'/'+filename), "a")
        for fname in examples: file_obj.write(fname + "\n")
        file_obj.close()

train_ds_repeated, train_ds_unique, val_ds = dataloader(train_files, val_files, args.epochs, args.batch_size, args.val_batch_size, args.augmentation, args.normalization)
train_steps_per_epoch = math.floor(len(train_files) // args.batch_size)

print('Loading val data')
val_steps_per_epoch = math.floor(len(val_files) // args.val_batch_size)
x_val_list = []
val_y_list = []

for y,k in val_ds:
    x_val_list.append(y)
    val_y_list.append(k)
x_val = tf.concat(x_val_list, axis=0)
val_y = tf.concat(val_y_list, axis=0)
val_y = val_y.numpy()

print('Loading train data')
x_train_list = []
train_y_list = []

for y,k in train_ds_unique:
    x_train_list.append(y)
    train_y_list.append(k)
x_train = tf.concat(x_train_list, axis=0)
train_y = tf.concat(train_y_list, axis=0)
train_y = train_y.numpy()

print('Done!')

# -------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------
# SET RANDOM SEED
random_seed = 42
tf.random.set_seed(random_seed)
np.random.seed(random_seed)

# -------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------
#MAKING DIRECTORIES
PROJECT_ROOT = Path.cwd()#os.getcwd()#

main_dir_str = 'models/'+args.model_name
main_dir = PROJECT_ROOT / main_dir_str
main_dir.mkdir(exist_ok=True)
model_dir_str = 'models/'+args.model_name+'/model'
model_dir = PROJECT_ROOT / model_dir_str
model_dir.mkdir(exist_ok=True)
# -------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------
#DEFINE MODELs
encoder = make_encoder_model(args.input_shape, args.z_dim)
decoder = make_decoder_model(args.input_shape, args.z_dim)
discriminator = make_discriminator_model(args.z_dim)

if args.load_weights_from is not 'None':
    encoder.load_weights(args.load_weights_from+'/encoder')
    decoder.load_weights('models/'+transfer_tag+'/model/decoder')
    discriminator.load_weights('models/'+transfer_tag+'/model/discriminator')
    
    for layer in encoder.layers[:args.freeze_layers[0]]:
        layer.trainable =  False
    for layer in decoder.layers[:args.freeze_layers[0]]:
        layer.trainable =  False
    for layer in discriminator.layers[:args.freeze_layers[0]]:
        layer.trainable =  False


# -------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------
# DEFINE OPTIMIZERS
ae_optimizer = tf.keras.optimizers.Adam(lr=args.learning_rate*args.learning_rate_factors[0], beta_1=args.adam_beta1, beta_2=args.adam_beta2)
dc_optimizer = tf.keras.optimizers.Adam(lr=args.learning_rate*args.learning_rate_factors[1],beta_1=args.adam_beta1, beta_2=args.adam_beta2)
gen_optimizer = tf.keras.optimizers.Adam(lr=args.learning_rate*args.learning_rate_factors[2], beta_1=args.adam_beta1, beta_2=args.adam_beta2)

# -------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------
# DEFINE TRAINING FUNCTIONS
@tf.function
def train_step(batch_x, ae_weight, dc_weight, gen_weight, z_dim):
    # -------------------------------------------------------------------------------------------------------------
    # Autoencoder
    with tf.GradientTape() as ae_tape:
        z_mean, z_std = encoder(batch_x[0], training=True)
        # Probabilistic with Gaussian posterior distribution
        epsilon = tf.random.normal(shape=z_mean.shape, mean=0, stddev=1)
        z = z_mean + (1e-8 + z_std) * epsilon
        decoder_output = decoder(z, training=True)
        ae_loss = ae_weight*autoencoder_loss(batch_x[0], decoder_output)
    ae_grads = ae_tape.gradient(ae_loss, encoder.trainable_variables + decoder.trainable_variables)
    ae_optimizer.apply_gradients(zip(ae_grads, encoder.trainable_variables + decoder.trainable_variables))

    # -------------------------------------------------------------------------------------------------------------
    # Discriminator 
    with tf.GradientTape() as dc_tape:
        real_distribution = tf.random.normal([batch_x[0].shape[0], args.z_dim], mean=0.0, stddev=0.2)
        z_mean, z_std = encoder(batch_x[0], training=True)
        epsilon = tf.random.normal(shape=z_mean.shape, mean=0, stddev=1)
        z = z_mean + (1e-8 + z_std) * epsilon
        dc_real = discriminator(real_distribution, training=True)
        dc_fake = discriminator(z, training=True)
        dc_loss = dc_weight*discriminator_loss(dc_real, dc_fake)
        dc_acc = accuracy(tf.concat([tf.ones_like(dc_real), tf.zeros_like(dc_fake)], axis=0),
                        tf.concat([dc_real, dc_fake], axis=0))
    dc_grads = dc_tape.gradient(dc_loss, discriminator.trainable_variables)
    dc_optimizer.apply_gradients(zip(dc_grads, discriminator.trainable_variables))
    # -------------------------------------------------------------------------------------------------------------
    # Generator (Encoder
    with tf.GradientTape() as gen_tape:
        z_mean, z_std = encoder(batch_x[0], training=True)
        # Probabilistic with Gaussian posterior distributio
        epsilon = tf.random.normal(shape=z_mean.shape, mean=0, stddev=1)
        z = z_mean + (1e-8 + z_std) * epsilon
        dc_fake = discriminator(z, training=True)
        # Generator loss
        gen_loss = gen_weight*generator_loss(dc_fake)
    gen_grads = gen_tape.gradient(gen_loss, encoder.trainable_variables)
    gen_optimizer.apply_gradients(zip(gen_grads, encoder.trainable_variables))
    return ae_loss, dc_loss, dc_acc, gen_loss
    # Training function

#-------------------------------------------------------------------------------------------------------------
# Validation Function
@tf.function
def val_step(batch_x, ae_weight, dc_weight, gen_weight, z_dim):
    z_mean, z_std = encoder(batch_x[0], training=True)

    # Probabilistic with Gaussian posterior distribution
    epsilon = tf.random.normal(shape=z_mean.shape, mean=0, stddev=1)
    z = z_mean + (1e-8 + z_std) * epsilon
    decoder_output = decoder(z, training=True)

    # Autoencoder loss
    ae_loss = ae_weight*autoencoder_loss(batch_x[0], decoder_output)

    real_distribution = tf.random.normal([batch_x[0].shape[0], args.z_dim], mean=0.0, stddev=1.0)
    z_mean, z_std = encoder(batch_x[0], training=True)

    # Probabilistic with Gaussian posterior distribution
    epsilon = tf.random.normal(shape=z_mean.shape, mean=0, stddev=1)
    z = z_mean + (1e-8 + z_std) * epsilon

    dc_real = discriminator(real_distribution, training=True)
    dc_fake = discriminator(z, training=True)

    # Discriminator Loss
    dc_loss = dc_weight*discriminator_loss(dc_real, dc_fake)

    # Discriminator Acc
    dc_acc = accuracy(tf.concat([tf.ones_like(dc_real), tf.zeros_like(dc_fake)], axis=0),
                    tf.concat([dc_real, dc_fake], axis=0))
    
    #Generator loss
    z_mean, z_std = encoder(batch_x[0], training=True)

    # Probabilistic with Gaussian posterior distribution
    epsilon = tf.random.normal(shape=z_mean.shape, mean=0, stddev=1)
    z = z_mean + (1e-8 + z_std) * epsilon

    dc_fake = discriminator(z, training=True)

    # Generator loss
    gen_loss = gen_weight*generator_loss(dc_fake)

    return ae_loss, dc_loss, dc_acc, gen_loss   

def scheduler(epoch, base_lr, k=0.0025, limit=1e-4, up=False):
    if up:
        lrate = limit - (limit-base_lr)*tf.math.exp(-k * epoch)
    else:
        lrate = base_lr * tf.math.exp(-k * epoch)
    return lrate

# -------------------------------------------------------------------------------------------------------------
# Training loop
print('Started training')
x_train_list = []
train_y_list = []
best_jds = 10000
accuracy = tf.keras.metrics.BinaryAccuracy()
fixed_noise_sampling = tf.random.normal(shape=(3*len(x_val),args.z_dim), mean=0, stddev=0.2)
fixed_noise_code = tf.random.normal(shape=(len(x_val),args.z_dim), mean=0, stddev=1.)

epoch = -1
fixed_noise = tf.random.normal(shape=(25,args.z_dim), mean=0, stddev=1)
for step, (batch_x) in enumerate(train_ds_repeated):
    if step % train_steps_per_epoch == 0:
        epoch = epoch + 1
        start = time.time()

        epoch_ae_loss_avg = tf.metrics.Mean()
        epoch_dc_loss_avg = tf.metrics.Mean()
        epoch_dc_acc_avg = tf.metrics.Mean()
        epoch_gen_loss_avg = tf.metrics.Mean()

        epoch_val_ae_loss_avg = tf.metrics.Mean()
        epoch_val_dc_loss_avg = tf.metrics.Mean()
        epoch_val_dc_acc_avg = tf.metrics.Mean()
        epoch_val_gen_loss_avg = tf.metrics.Mean()

        ## Should we keep this? honestly have no memory if its working correctly or not
        if DECAY:
            if epoch>=reg_start:
                ae_optimizer.lr = scheduler(epoch-reg_start, args.learning_rate*args.learning_rate_factors[0], decay_rate,limit=args.learning_rate/args.learning_rate_factors[0], up=False)
                dc_optimizer.lr = scheduler(epoch-reg_start, args.learning_rate*args.learning_rate_factors[1], growth_rate, limit=args.learning_rate/args.learning_rate_factors[1], up=True)
                gen_optimizer.lr = scheduler(epoch-reg_start, args.learning_rate*args.learning_rate_factors[2], growth_rate, limit=args.learning_rate/args.learning_rate_factors[2], up=True)

    ae_loss, dc_loss, dc_acc, gen_loss = train_step(batch_x, args.weights_losses[0], args.weights_losses[1], args.weights_losses[2], args.z_dim)

    epoch_ae_loss_avg(ae_loss)
    epoch_dc_loss_avg(dc_loss)
    epoch_dc_acc_avg(dc_acc)
    epoch_gen_loss_avg(gen_loss)

    if step % train_steps_per_epoch == 0:
        epoch_time = time.time() - start
        print('TRAINING {:4d}: TIME: {:.2f} ETA: {:.2f} AE_LOSS: {:.4f} DC_LOSS: {:.4f} DC_ACC: {:.4f} GEN_LOSS: {:.4f}' \
            .format(epoch, epoch_time,
                    epoch_time * (args.epochs - epoch),
                    epoch_ae_loss_avg.result(),
                    epoch_dc_loss_avg.result(),
                    epoch_dc_acc_avg.result(),
                    epoch_gen_loss_avg.result()))

        
        for batch, (batch_x) in enumerate(val_ds):
            ae_val_loss, dc_val_loss, dc_val_acc, gen_val_loss = val_step(batch_x, args.weights_losses[0], args.weights_losses[1], args.weights_losses[2], args.z_dim)

            epoch_val_ae_loss_avg(ae_val_loss)
            epoch_val_dc_loss_avg(dc_val_loss)
            epoch_val_dc_acc_avg(dc_val_acc)
            epoch_val_gen_loss_avg(gen_val_loss)

            epoch_time = time.time() - start
            print('VALIDATION {:4d}: TIME: {:.2f} ETA: {:.2f} AE_LOSS: {:.4f} DC_LOSS: {:.4f} DC_ACC: {:.4f} GEN_LOSS: {:.4f}' \
                .format(epoch, epoch_time,
                        epoch_time * (args.epochs - epoch),
                        epoch_val_ae_loss_avg.result(),
                        epoch_val_dc_loss_avg.result(),
                        epoch_val_dc_acc_avg.result(),
                        epoch_val_gen_loss_avg.result()))

        # -------------------------------------------------------------------------------------------------------------
        if epoch % args.save_frequency == 0:
            sampled = []
            for i in range(len(fixed_noise_sampling)):
                z = np.reshape(fixed_noise_sampling[i,:], (1, args.z_dim))
                x = decoder(z, training=False).numpy().squeeze()
                sampled.append(x)
                
            jsd = jsd_between_point_cloud_sets(x_val, np.asarray(sampled))
            if jsd < best_jds:
                encoder.save_weights(model_dir_str+'/encoder')
                decoder.save_weights(model_dir_str+'/decoder')
                discriminator.save_weights(model_dir_str+'/discriminator')
                best_jds = jsd
                 
print('Training finished')
