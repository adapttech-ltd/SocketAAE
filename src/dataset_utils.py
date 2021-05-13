from glob import glob
import numpy as np
import open3d as o3d
import tensorflow as tf
import copy
from sklearn.model_selection import train_test_split


def tf_parse_filename(filename, normalization='None', normalization_factor=1, augmentation=False):
    """Take batch of filenames and create point cloud and label"""

    idx_lookup = {'airplane': 0, 'bathtub': 1, 'bed': 2, 'bench': 3, 'bookshelf': 4,
                  'bottle': 5, 'bowl': 6, 'car': 7, 'chair': 8, 'cone': 9,
                  'cup': 10, 'curtain': 11, 'desk': 12, 'door': 13, 'dresser': 14,
                  'flower_pot': 15, 'glass_box': 16, 'guitar': 17, 'keyboard': 18,
                  'lamp': 19, 'laptop': 20, 'mantel': 21, 'monitor': 22, 'night_stand': 23,
                  'person': 24, 'piano': 25, 'plant': 26, 'radio': 27, 'range_hood': 28,
                  'sink': 29, 'sofa': 30, 'stairs': 31, 'stool': 32, 'table': 33,
                  'tent': 34, 'toilet': 35, 'tv_stand': 36, 'vase': 37, 'wardrobe': 38,
                  'xbox': 39, 'sockettt':0, 'sockettf': 1}

    def parse_filename(filename_batch, normalization='None', normalization_factor=1, augmentation=False):
        pt_clouds = []
        #labels = []
        for filename in filename_batch:
            if tf.strings.split(filename, '.')[-1].numpy().decode() == 'npy':
                # Read in point cloud
                filename_str = filename.numpy().decode()
                pt_cloud = np.load(filename_str)
            else:
                filename_str = filename.numpy().decode()
                pc = o3d.io.read_point_cloud(filename_str)
                pt_cloud = np.asarray(pc.points)
            center = [np.mean(pt_cloud[:,0]), np.mean(pt_cloud[:,1]), np.mean(pt_cloud[:,2])]
            pt_cloud = pt_cloud - np.asarray(center)
            # Add rotation and jitter to point cloud
            if augmentation:
                theta = np.random.random() * 2*3.141
                A = np.array([[np.cos(theta), -np.sin(theta), 0],
                            [np.sin(theta), np.cos(theta), 0],
                            [0, 0, 1]])
                offsets = np.random.normal(0, 0.02, size=pt_cloud.shape)
                pt_cloud = np.matmul(pt_cloud, A) + offsets

            pt_clouds.append(pt_cloud)
            #labels.append(label)
        max_individual = np.asarray([np.max(np.linalg.norm(pc, axis=1)) for pc in pt_clouds])
        if normalization.numpy():
            pt_clouds = [pc/max_individual[i] for i,pc in enumerate(pt_clouds)]
        return np.stack(pt_clouds)#, np.stack(labels)

    x = tf.py_function(parse_filename, [filename, normalization,  normalization_factor, augmentation], tf.float32)
    x.set_shape([None for _ in range(3)])
    return x, x


def tf_parse_filename_classes(filename, normalization='None', normalization_factor=1, augmentation=False):
    """Take batch of filenames and create point cloud and label"""

    idx_lookup = {'airplane': 0, 'bathtub': 1, 'bed': 2, 'bench': 3, 'bookshelf': 4,
                  'bottle': 5, 'bowl': 6, 'car': 7, 'chair': 8, 'cone': 9,
                  'cup': 10, 'curtain': 11, 'desk': 12, 'door': 13, 'dresser': 14,
                  'flower_pot': 15, 'glass_box': 16, 'guitar': 17, 'keyboard': 18,
                  'lamp': 19, 'laptop': 20, 'mantel': 21, 'monitor': 22, 'night_stand': 23,
                  'person': 24, 'piano': 25, 'plant': 26, 'radio': 27, 'range_hood': 28,
                  'sink': 29, 'sofa': 30, 'stairs': 31, 'stool': 32, 'table': 33,
                  'tent': 34, 'toilet': 35, 'tv_stand': 36, 'vase': 37, 'wardrobe': 38,
                  'xbox': 39, 'sockettt':0, 'sockettf': 1, 'can':2, 'tin_can':3, 'mug':4, 'jar':5, 'AC1':0, 'AC5_2':1,
                  'AC6_2':2, 'AC8_20200518':3, 'AC9':4, 'AC12':5}

    def parse_filename_classes(filename_batch, normalization='None', normalization_factor=1, augmentation=False):
        pt_clouds = []
        labels = []
        inds = []
        pt_cloud_no_outliers = np.asarray([])
        for filename in filename_batch:
            inds = []
            if tf.strings.split(filename, '.')[-1].numpy().decode() == 'npy':
                # Read in point cloud
                filename_str = filename.numpy().decode()
                pt_cloud = np.load(filename_str)
            else:
                filename_str = filename.numpy().decode()
                pc = o3d.io.read_point_cloud(filename_str)
                pt_cloud = np.asarray(pc.points)
                pt_cloud = np.asarray(pc.points)
                #inds.extend(abs(pt_cloud[:,2] - np.mean(pt_cloud[:,2])) > 0.008 * np.std(pt_cloud[:,2]))
                #inds.extend(abs(pt_cloud[:,1] - np.mean(pt_cloud[:,1])) > 0.008 * np.std(pt_cloud[:,1]))
                #inds.extend(abs(pt_cloud[:,0] - np.mean(pt_cloud[:,0])) > 0.008 * np.std(pt_cloud[:,0]))
                #inds = np.unique(np.asarray(inds))
                #print(len(inds))
                #pt_cloud_no_outliers = np.asarray([pt_cloud[i] for i in range(len(pt_cloud)) if i not in inds])
                #_, inds = o3d.geometry.PointCloud().remove_statistical_outlier(20, 0.2)
                #o3d.visualization.draw_geometries([pc_no_outliers])

                #pt_cloud_no_outliers = np.asarray(pc.points)[inds]
            #center = [np.mean(pt_cloud_no_outliers[:,0]), np.mean(pt_cloud_no_outliers[:,1]), np.mean(pt_cloud_no_outliers[:,2])]
            #pt_cloud = pt_cloud - np.asarray(center)
            center = [np.mean(pt_cloud[:,0]), np.mean(pt_cloud[:,1]), np.mean(pt_cloud[:,2])]
            pt_cloud = pt_cloud - np.asarray(center)
            #inds.extend(np.argwhere(abs(pt_cloud[:,2] - np.mean(pt_cloud[:,2])) > 2 * np.std(pt_cloud[:,2])))
            #inds.extend(np.argwhere(abs(pt_cloud[:,1] - np.mean(pt_cloud[:,1])) > 2 * np.std(pt_cloud[:,1])))
            #inds.extend(np.argwhere(abs(pt_cloud[:,0] - np.mean(pt_cloud[:,0])) > 2 * np.std(pt_cloud[:,0])))
            #inds = np.unique(np.asarray(inds))
            #pt_cloud_no_outliers = np.asarray([pt_cloud[i] for i in range(len(pt_cloud)) if i not in inds])
            #tf.print(inds.shape)
            #dists = np.linalg.norm(pt_cloud, axis=1)
            if normalization=='Single':
                #old_range = (np.max(np.linalg.norm(pt_cloud, axis=1))- np.min(np.linalg.norm(pt_cloud, axis=1)))
                #new_range = 1
                #pt_cloud = ((pt_cloud - np.min(np.linalg.norm(pt_cloud, axis=1)))/old_range) + 0.5
                pt_cloud = pt_cloud/np.max(np.linalg.norm(pt_cloud, axis=1))#pt_cloud_no_outliers, axis=1))
            # Add rotation and jitter to point cloud
            if augmentation:
                theta = np.random.random() * 2*3.141
                A = np.array([[np.cos(theta), -np.sin(theta), 0],
                            [np.sin(theta), np.cos(theta), 0],
                            [0, 0, 1]])
                offsets = np.random.normal(0, 0.01, size=pt_cloud.shape)
                pt_cloud = np.matmul(pt_cloud, A) + offsets
            # Create classification label
            obj_type = filename_str.split('/')[1]# e.g., airplane, bathtub
            #label = np.zeros(40, dtype=np.float32)
            #label[idx_lookup[obj_type]] = 1.0
            label = idx_lookup[obj_type]
            labels.append(label)
            pt_clouds.append(pt_cloud)
        #max_individual = np.asarray([np.max(np.linalg.norm(pc, axis=1)) for pc in pt_clouds])
        #if normalization.numpy().decode()=='Single':
        #    pt_clouds = [pc/max_individual[i] for i,pc in enumerate(pt_clouds)]
        #elif normalization.numpy().decode()=='Relative':
        #    pt_clouds = [pc/normalization_factor for i,pc in enumerate(pt_clouds)] 
        return np.stack(pt_clouds), np.stack(labels)

    x,y = tf.py_function(parse_filename_classes, [filename, normalization,  normalization_factor, augmentation], [tf.float32, tf.int32])
    x.set_shape([None for _ in range(3)])
    y.set_shape([None for _ in range(1)])
    return x, y


def tf_parse_filename_noise(filename, normalization='None', normalization_factor=1, augmentation=False):
    """Take batch of filenames and create point cloud and label"""

    idx_lookup = {'airplane': 0, 'bathtub': 1, 'bed': 2, 'bench': 3, 'bookshelf': 4,
                  'bottle': 5, 'bowl': 6, 'car': 7, 'chair': 8, 'cone': 9,
                  'cup': 10, 'curtain': 11, 'desk': 12, 'door': 13, 'dresser': 14,
                  'flower_pot': 15, 'glass_box': 16, 'guitar': 17, 'keyboard': 18,
                  'lamp': 19, 'laptop': 20, 'mantel': 21, 'monitor': 22, 'night_stand': 23,
                  'person': 24, 'piano': 25, 'plant': 26, 'radio': 27, 'range_hood': 28,
                  'sink': 29, 'sofa': 30, 'stairs': 31, 'stool': 32, 'table': 33,
                  'tent': 34, 'toilet': 35, 'tv_stand': 36, 'vase': 37, 'wardrobe': 38,
                  'xbox': 39, 'sockettt':0, 'sockettf': 1, 'can':2, 'tin_can':3, 'mug':4, 'jar':5, 'AC1':0, 'AC5_2':1,
                  'AC6_2':2, 'AC8_20200518':3, 'AC9':4, 'AC12':5}

    def parse_filename_noise(filename_batch, normalization='None', normalization_factor=1, augmentation=False):
        pt_clouds = []
        labels = []
        for filename in filename_batch:
            if tf.strings.split(filename, '.')[-1].numpy().decode() == 'npy':
                # Read in point cloud
                filename_str = filename.numpy().decode()
                pt_cloud = np.load(filename_str)
            else:
                filename_str = filename.numpy().decode()
                pc = o3d.io.read_point_cloud(filename_str)
                #pc_no_outliers = pc.remove_statistical_outliers(20, 0.95)
                #o3d.visualization.draw_geometries([pc_no_outliers])
                pt_cloud = np.asarray(pc.points)
                #pt_cloud_no_outliers = np.asarray(pc_no_outliers.points)
            center = [np.mean(pt_cloud[:,0]), np.mean(pt_cloud[:,1]), np.mean(pt_cloud[:,2])]
            pt_cloud = pt_cloud - np.asarray(center)#
            #pt_cloud_no_outliers = pt_cloud_no_outliers - np.asarray(center)
            if normalization=='Single':
                #old_range = (np.max(np.linalg.norm(pt_cloud, axis=1))- np.min(np.linalg.norm(pt_cloud, axis=1)))
                #new_range = 1
                #pt_cloud = ((pt_cloud - np.min(np.linalg.norm(pt_cloud, axis=1)))/old_range) + 0.5
                pt_cloud = pt_cloud/np.max(np.linalg.norm(pt_cloud, axis=1))
            # Add rotation and jitter to point cloud
            pt_cloud_or = copy.deepcopy(pt_cloud)
            if augmentation:
                theta = np.random.random() * 2*3.141
                #A = np.array([[np.cos(theta), -np.sin(theta), 0],
                #            [np.sin(theta), np.cos(theta), 0],
                #            [0, 0, 1]])
                offsets = np.random.normal(0, 0.05, size=pt_cloud.shape)
                pt_cloud = pt_cloud + offsets #np.matmul(pt_cloud, A) +
            # Create classification label
            #obj_type = filename_str.split('/')[1]# e.g., airplane, bathtub
            #label = np.zeros(40, dtype=np.float32)
            #label[idx_lookup[obj_type]] = 1.0
            #label = pt_cloud_or
            labels.append(pt_cloud_or)
            pt_clouds.append(pt_cloud)
        #max_individual = np.asarray([np.max(np.linalg.norm(pc, axis=1)) for pc in pt_clouds])
        #if normalization.numpy().decode()=='Single':
        #    pt_clouds = [pc/max_individual[i] for i,pc in enumerate(pt_clouds)]
        #elif normalization.numpy().decode()=='Relative':
        #    pt_clouds = [pc/normalization_factor for i,pc in enumerate(pt_clouds)] 
        return np.stack(pt_clouds), np.stack(labels)

    x,y = tf.py_function(parse_filename_noise, [filename, normalization,  normalization_factor, augmentation], [tf.float32, tf.float32])
    x.set_shape([None for _ in range(3)])
    y.set_shape([None for _ in range(3)])
    return x, y


def load_filenames(dataset):
    files, labels = [], []
    if dataset=='ShapeNet':
        string = [dataset+'/*/']
        termination = '*.ply'
    elif dataset=='Paraboloids':
        string = [dataset+'/*/']
        termination = ['*.ply', '*.npy']
    elif dataset == 'LaserDatasetAug' or dataset == 'LaserDataset2048':
        string = [dataset+'/*/']
        termination = '*.npy'
    else:
        string = [dataset+'/*/']
        termination = 'train/*.npy'
    for s in string:
        for obj_type in glob(s):
            cur_files=glob(obj_type + termination)
            print(type(cur_files[0]))
            files.extend(cur_files)
            labels.extend(np.asarray([cur_files[i].split('/')[1] for i in range(len(cur_files))]))

    return files, labels

def split_files(dataset_path, termination, train_split, val_split, test_split):
    test_files = []
    train_val_files = []
    val_files = []
    files_dir = dataset_path+'/*.'+termination
    files = glob(files_dir)
    print(len(files))
    train_val, test = train_test_split(files, train_size=train_split+val_split, random_state=0, shuffle=True)
    test_files.extend(test)
    train_val_files.extend(train_val)
    if val_split > 0:
        print(train_split, val_split, test_split)
        print(len(train_val_files))
        train_files, val_files = train_test_split(train_val_files, train_size=train_split/(1-test_split), random_state=0)
    else:
        train_files = train_val_files
    return train_files, val_files, test_files


def train_val_split(train_size=0.80, dataset='ModelNet40', files=[]):
    train, val = [], []
    if len(files)==0:
        if dataset=='ShapeNet':
            string = [dataset+'/*/']
            termination = '*.ply'
        elif dataset=='Paraboloids':
            string = [dataset+'/*/']
            termination = ['*.ply', '*.npy']
        elif dataset == 'LaserDatasetAug' or dataset == 'LaserDataset2048':
            string = [dataset+'/*/']
            termination = '*.npy'
        else:
            string = [dataset+'/*/']
            termination = 'train/*.npy'
        for s in string:
            for obj_type in glob(s):
                #cur_files = []
               # for term in termination:
                cur_files=glob(obj_type + termination)
                cur_train, cur_val = \
                    train_test_split(cur_files, train_size=train_size, random_state=0, shuffle=True)
                train.extend(cur_train)
                val.extend(cur_val)
    else:
        train, val = \
                    train_test_split(files, train_size=train_size, random_state=0, shuffle=True)
    return train, val

def load_termination(path, termination='noisy*.ply'):
    string = [path+'/*/']
    filenames =  []
    for s in string:
        for obj_type in glob(s):
            files = glob(obj_type+termination)
            filenames.extend(files)
    return filenames



def train_val_split_classes(train_size=0.92, dataset='ModelNet40', classes=[], files=[]):
    train, val = [], []
    if len(files)==0:
        if dataset=='ShapeNet':
            string = ['ShapeNet/'+clas+'/' for clas in classes]
            termination = '*.ply'
        else:
            string = [dataset+'/'+clas+'/' for clas in classes]
            termination = 'train/*.npy'
        for s in string:
            for obj_type in glob(s):
                cur_files = glob(obj_type + termination)
                cur_train, cur_val = train_test_split(cur_files, train_size=train_size, random_state=0, shuffle=True)
                train.extend(cur_train)
                val.extend(cur_val)
    else:
        train, val = \
                    train_test_split(files, train_size=train_size, random_state=0, shuffle=True)
    return train, val

def train_val_split_single_class(train_size=0.90, dataset='ShapeNet/02691156', files=[]):
    train, val = [], []
    if len(files)==0:
        if dataset.split('/')[0]=='ShapeNet':
            termination = '/*.ply'
        else:
            termination = '/train/*.npy'

        for obj_type in glob(dataset):
                cur_files = glob(obj_type + termination)
                cur_train, cur_val = \
                    train_test_split(cur_files, train_size=train_size, random_state=0, shuffle=True)
                train.extend(cur_train)
                val.extend(cur_val)
    else:
        train, val = \
                    train_test_split(files, train_size=train_size, random_state=0, shuffle=True)

    return train, val

def dataloader(train_files, val_files, epochs=200, train_batch_size=32, val_batch_size=32, augmentation=True, normalization=0, normalization_factor=1):
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    train_ds = tf.data.Dataset.list_files(train_files)
    train_ds = train_ds.batch(train_batch_size, drop_remainder=False)
    train_ds_unrepeated = train_ds.map(lambda x: tf_parse_filename(x,normalization=normalization, normalization_factor=normalization_factor, augmentation=augmentation))#, num_parallel_calls=AUTOTUNE)
    train_ds_unrepeated = train_ds_unrepeated.prefetch(buffer_size=AUTOTUNE)

    train_ds = train_ds.repeat(epochs)
    train_ds = train_ds.map(lambda x: tf_parse_filename(x,normalization=normalization, normalization_factor=normalization_factor, augmentation=augmentation))#, num_parallel_calls=AUTOTUNE)
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)

    val_ds = tf.data.Dataset.list_files(val_files)
    val_ds = val_ds.batch(val_batch_size, drop_remainder=True)
    val_ds_unrepeated = val_ds.map(lambda x: tf_parse_filename(x,normalization=normalization, normalization_factor=normalization_factor, augmentation=augmentation))#, num_parallel_calls=AUTOTUNE)
    val_ds_unrepeated = val_ds_unrepeated.prefetch(buffer_size=AUTOTUNE)

    return train_ds, train_ds_unrepeated, val_ds_unrepeated

def test_dataloader(test_files, normalization, normalization_factor=1):
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    test_ds = tf.data.Dataset.list_files(test_files)
    test_ds = test_ds.batch(1)
    test_ds = test_ds.map(lambda x: tf_parse_filename(x,normalization=normalization, normalization_factor=normalization_factor, augmentation=0))#, num_parallel_calls=AUTOTUNE)
    test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)
    return test_ds

def dataloader_w_classes(train_files, val_files, repeat=True, epochs=2000, drop=True, train_batch_size=32, val_batch_size=32, augmentation=True, normalization='None', normalization_factor=1):
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    train_ds = tf.data.Dataset.list_files(train_files, shuffle=True)
    train_ds2 = train_ds.batch(train_batch_size, drop_remainder=drop)
    if repeat:
        train_ds2 = train_ds2.repeat(epochs)
    train_ds_unrepeated = train_ds2.map(lambda x: tf_parse_filename_classes(x,normalization=normalization, normalization_factor=normalization_factor, augmentation=augmentation))#, num_parallel_calls=AUTOTUNE)
    if not repeat:
        train_ds_unrepeated = train_ds_unrepeated.repeat(epochs)
    train_ds_unrepeated = train_ds_unrepeated.prefetch(buffer_size=AUTOTUNE)
    train_ds_unique = train_ds.batch(train_batch_size, drop_remainder=drop)
    train_ds_unique = train_ds_unique.map(lambda x: tf_parse_filename_classes(x,normalization=normalization, normalization_factor=normalization_factor, augmentation=augmentation))#, num_parallel_calls=AUTOTUNE)
    train_ds_unique = train_ds_unique.prefetch(buffer_size=AUTOTUNE)

    val_ds_unrepeated = []
    if len(val_files)>0:
        val_ds = tf.data.Dataset.list_files(val_files, shuffle=True)
        val_ds = val_ds.batch(val_batch_size, drop_remainder=drop)
        val_ds_unrepeated = val_ds.map(lambda x: tf_parse_filename_classes(x,normalization=normalization, normalization_factor=normalization_factor, augmentation=augmentation))#, num_parallel_calls=AUTOTUNE)
        val_ds_unrepeated = val_ds_unrepeated.prefetch(buffer_size=AUTOTUNE)

    return train_ds_unrepeated, train_ds_unique, val_ds_unrepeated

def test_dataloader_w_classes(test_files, batch_size=1, augmentation=True, normalization='Single', normalization_factor=1):
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    test_ds = tf.data.Dataset.list_files(test_files)
    test_ds = test_ds.batch(batch_size, drop_remainder=True)
    test_ds = test_ds.map(lambda x: tf_parse_filename_classes(x,normalization=normalization, normalization_factor=normalization_factor, augmentation=augmentation))#, num_parallel_calls=AUTOTUNE)
    test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)
    return test_ds



def noise_dataloader(train_files, val_files, repeat=True, epochs=2000, drop=True, train_batch_size=32, val_batch_size=32, augmentation=True, normalization='None', normalization_factor=1):
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    train_ds = tf.data.Dataset.list_files(train_files, shuffle=False)
    train_ds2 = train_ds.batch(train_batch_size, drop_remainder=drop)
    if repeat:
        train_ds2 = train_ds2.repeat(epochs)
    train_ds_unrepeated = train_ds2.map(lambda x: tf_parse_filename_noise(x,normalization=normalization, normalization_factor=normalization_factor, augmentation=augmentation))#, num_parallel_calls=AUTOTUNE)
    if not repeat:
        train_ds_unrepeated = train_ds_unrepeated.repeat(epochs)
    train_ds_unrepeated = train_ds_unrepeated.prefetch(buffer_size=AUTOTUNE)
    train_ds_unique = train_ds.batch(train_batch_size, drop_remainder=drop)
    train_ds_unique = train_ds_unique.map(lambda x: tf_parse_filename_noise(x,normalization=normalization, normalization_factor=normalization_factor, augmentation=augmentation))#, num_parallel_calls=AUTOTUNE)
    train_ds_unique = train_ds_unique.prefetch(buffer_size=AUTOTUNE)
    #train_ds = train_ds.repeat(epochs)
    #train_ds = train_ds.map(lambda x: tf_parse_filename_classes(x,normalization=normalization, normalization_factor=normalization_factor, augmentation=augmentation))#, num_parallel_calls=AUTOTUNE)
    #train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)

    val_ds = tf.data.Dataset.list_files(val_files, shuffle=False)
    val_ds = val_ds.batch(val_batch_size, drop_remainder=drop)
    val_ds_unrepeated = val_ds.map(lambda x: tf_parse_filename_noise(x,normalization=normalization, normalization_factor=normalization_factor, augmentation=augmentation))#, num_parallel_calls=AUTOTUNE)
    val_ds_unrepeated = val_ds_unrepeated.prefetch(buffer_size=AUTOTUNE)
    return train_ds_unrepeated, train_ds_unique, val_ds_unrepeated


def test_dataloader_noise(test_files, batch_size=1, augmentation=True, normalization='Single', normalization_factor=1):
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    test_ds = tf.data.Dataset.list_files(test_files)
    test_ds = test_ds.batch(batch_size, drop_remainder=True)
    test_ds = test_ds.map(lambda x: tf_parse_filename_noise(x,normalization=normalization, normalization_factor=normalization_factor, augmentation=augmentation))#, num_parallel_calls=AUTOTUNE)
    test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)
    return test_ds
