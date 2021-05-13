import point_cloud_utils as pcu
import glob
import numpy as np
import open3d as open3d
import matplotlib.pylab  as plt
import re
import os
import yaml
import sys
import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('--dataset_path', '-dp', type=str, required=True,
                    help='path to data folder')
parser.add_argument('--save_path', '-sp',type=str, default='None',
                    help='path to save resampled files')
parser.add_argument('--n_points','-n', type=int, default=2048,
                    help='number of points to resample to')
parser.add_argument('--termination', '-t', default='.off',
                    help='termination to search for in folder, .off or .obj')

args = parser.parse_args()

def files_in_subdirs(top_dir, search_pattern):
    for path, _, files in os.walk(top_dir):
        for name in files:
            full_name = os.path.join(path, name)
            if full_name.endswith(search_pattern):
                yield full_name

filenames = [f for f in files_in_subdirs(args.dataset_path, args.termination)]
for i, fi in enumerate(filenames):
    path = os.path.split(fi)[0]
    foldername = path.replace(args.dataset_path+'/','')
    name = os.path.split(fi)[-1].split('.')[0]
    if args.save_path=='None':
        args.save_path = os.path.split(args.dataset_path)[0]+'/'+os.path.split(args.dataset_path)[1]+'_resampled'

    if not os.path.exists(args.save_path): os.makedirs(args.save_path)
    if os.path.split(foldername)[-1] == args.dataset_path.split('/')[-1]: #Single folder structure
        destination_filename = args.save_path+'/'+name
    else:
        if not os.path.exists(args.save_path+'/'+foldername): os.makedirs(args.save_path+'/'+foldername)
        destination_filename = args.save_path+'/'+foldername+'/'+name
    
    if args.termination=='.off':
        v, f, n = pcu.read_off(fi)
    elif args.termination=='.obj':
        v, f, n = pcu.read_obj(fi)
    else:
        print('Invalid termination')
        sys.exit(1)
    if len(f)!=0:
        samples = pcu.sample_mesh_lloyd(v, f, args.n_points) #normals inside v, poorly saved
        np.save(destination_filename+'.npy', samples)  
