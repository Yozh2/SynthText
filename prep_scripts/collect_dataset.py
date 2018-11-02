# system imports
import os
import os.path as osp
import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm

# define default paths and constants
VERBOSE = False
COLLECT_IMAGES_PATH = '../data/images/raw'
COLLECT_DEPTHS_PATH = '../data/images/depths/depths.h5'
COLLECT_LABELS_PATH = '../data/images/labels/labels.h5'
COLLECT_OUTPUT_PATH = '../data/dset.h5'

def print_attrs(name, obj):
    '''Print dataset as a tree'''
    print(name)
    for key, val in obj.attrs.items():
        print("    %s: %s" % (key, val))

def check_path(path):
    if not osp.exists(path):
        print(f'[COLLECT]: input {path} was not found!')
        raise FileNotFoundError
        return False
    else:
        return True

def collect_existing_datasets(path_images=COLLECT_IMAGES_PATH,
                              path_depths=COLLECT_DEPTHS_PATH,
                              path_labels=COLLECT_LABELS_PATH,
                              path_output=COLLECT_OUTPUT_PATH):
    '''Collect everything into one single dataset'''

    # Find images
    check_path(path_images)
    images_names = sorted([f for f in os.listdir(path_images) if osp.isfile(osp.join(path_images, f))])
    print(f'[COLLECT]: Found {len(images_names)} images in {path_images}')

    # Open depth dataset
    check_path(path_depths)
    depths = h5py.File(osp.abspath(path_depths), 'r')
    depths_names = sorted([n[0].decode() for n in list(depths['names'])])
    print(f'[COLLECT]: Found {len(depths_names)} depths in {path_depths}')

    # Open labels dataset
    check_path(path_labels)
    labels = h5py.File(osp.abspath(path_labels), 'r')
    labels_names = sorted([n for n in list(labels['mask'])])
    print(f'[COLELCT]: Found {len(labels_names)} labels in {path_labels}')

    # Check if we got the same amount of images, depths, labels
    print('[COLLECT]: Matching image names with depth and label names')
    assert len(images_names) == len(depths_names) == len(labels_names)
    for i in range(len(images_names)):
        assert images_names[i] == depths_names[i] == labels_names[i]
    print('[COLLECT]: Seems legit')

    # Create output dataset
    if not osp.exists(path_output):
        print(f'[COLLECT]: Storing data in {path_output}')
        os.makedirs(osp.dirname(path_output))

    out_dset = h5py.File(osp.abspath(path_output), 'w')
    out_dset.create_group('depth')
    out_dset.create_group('image')
    out_dset.create_group('seg')

    # process every image
    for imname in tqdm(images_names, desc='[COLLECT]: Collecting data'):

        # Load and preprocess data
        image = np.asarray(Image.open(osp.join(path_images, imname)))
        depth = np.squeeze(depths['depths'][imname])
        label = labels['mask'][imname][:]

        # Store data
        out_dset['image'].create_dataset(imname, data=image)
        out_dset['depth'].create_dataset(imname, data=depth)
        out_dset['seg'].create_dataset(imname, data=label)

        # Store segmentation attributes (areas and labels)
        for key in labels['mask'][imname].attrs.keys():
            out_dset['seg'][imname].attrs[key] = labels['mask'][imname].attrs[key].copy()

    out_dset.close()
    print('[COLLECT]: Done')

if __name__ == '__main__':
    collect_existing_datasets()
