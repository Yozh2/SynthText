"""
Python script to "flood-fill" the segments computed using gPb-UCM.
This assings the same integer label to all the pixels in the same segment.

Author: Ankush Gupta
"""

import argparse
import os
import os.path as osp
import cv2
import h5py
import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np
import scipy.io as sio
import sys
import traceback
from tqdm import tqdm

# Define default paths
MY_DIR = osp.dirname(osp.abspath(__file__))
BASE_DIR = osp.join(MY_DIR, '../data/images')
DB_PATH = osp.join(BASE_DIR, 'segs', 'segs.h5')
OUT_PATH = osp.join(BASE_DIR, 'labels', 'labels.h5')

def get_seed(sx,sy,ucm):
    n = sx.size
    for i in range(n):
        if ucm[sx[i]+1,sy[i]+1] == 0:
            return (sy[i],sx[i])

def get_mask(ucm, viz=False):
    ucm = ucm.copy()
    h,w = ucm.shape[:2]
    mask = np.zeros((h-2, w-2), 'float32')

    i = 0
    sx,sy = np.where(mask==0)
    seed = get_seed(sx, sy, ucm)
    areas = []
    labels=[]
    while seed is not None and i<1000:
        cv2.floodFill(mask, ucm, seed, i + 1)
        # calculate the area (no. of pixels):
        areas.append(np.sum(mask == i + 1))
        labels.append(i + 1)

        # get the location of the next seed:
        sx,sy = np.where(mask == 0)
        seed = get_seed(sx, sy, ucm)
        i += 1
    # print("[floodFill]: floodfill terminated in %d steps" % i)

    if viz:
        plt.imshow(mask)
        plt.show()

    return mask,np.array(areas),np.array(labels)

def get_mask_parallel(ucm_imname):
    ucm, imname = ucm_imname
    try:
        return (get_mask(ucm.T),imname)
    except:
        traceback.print_exc(file=sys.stdout)
        return None

def process_db_parallel(db_path, dbo, th=0.11):
    """
    Get segmentation masks from gPb contours.
    """
    class ucm_iterable(object):
        def __init__(self, ucm_path, th):
            self.th = th
            self.ucm_h5 = h5py.File(db_path,'r')
            self.N = self.ucm_h5['names'].size
            # print(list(self.ucm_h5['names']))
            self.i = 0

        def __iter__(self):
            return self

        def get_imname(self, i=None):
            # return "".join(map(chr, self.ucm_h5['names'][0,self.i][:]))
            return self.ucm_h5['names'][i][0].decode()

        def __stop__(self):
            # print("[floodFill]: Done iterations")
            self.ucm_h5.close()
            raise StopIteration

        def get_valid_name(self):
            if self.i >= self.N:
                self.__stop__()

            imname = self.get_imname(self.i)
            # print(f"i: {self.i}, imname: {imname}, len: {len(imname)}")
            while self.i < self.N-1 and len(imname) < 4:
                self.i += 1
                imname = self.get_imname(self.i)

            if len(imname) < 4:
               self.__stop__()

            # print(f'return valid {imname}')
            return imname

        def __next__(self):
            imname = self.get_valid_name()
            # print(f"[floodFill]: {self.i + 1} of {self.N}")
            # print(f'getting ucm for {imname}')
            ucm = self.ucm_h5['ucms'][imname][:]
            ucm = ucm.copy()
            self.i += 1
            return ((ucm > self.th).astype(np.uint8), imname)

    ucm_iter = ucm_iterable(db_path, th)
    cpu_count = mp.cpu_count()
    print("[floodFill]: cpu count: ", cpu_count)
    parpool = mp.Pool(cpu_count)

    print('[floodFill]: Creating multiprocessing pool')
    ucm_result = parpool.imap_unordered(get_mask_parallel, ucm_iter, chunksize=1)

    names = list()
    with tqdm(total=ucm_iter.N, desc='[floodFill]: Processing data') as pbar:
        for res in ucm_result:
            if res is None:
                continue
            
            ((mask,area,label), imname) = res
            # print("[floodFill]: got back: ", imname)
            mask = mask.astype('uint16')
            mask_dset = dbo['mask'].create_dataset(imname, data=mask.T)
            names.append(imname)
            mask_dset.attrs['area'] = area
            mask_dset.attrs['label'] = label

            pbar.update()


def floodfill_dataset(db_path=DB_PATH, out_path=OUT_PATH, verbose=False):

    if not osp.exists(db_path):
        if verbose:
            print('[floodFill]: Input dataset not found', db_path)
        raise FileNotFoundError

    if verbose:
        print('[floodFill]: Creating output h5 file %s' % out_path)

    if not osp.exists(osp.dirname(out_path)):
        os.makedirs(osp.dirname(out_path), exist_ok=True)



    dbo = h5py.File(out_path, 'w')
    dbo.create_group("mask")

    process_db_parallel(db_path, dbo)
    dbo.close()

    if verbose:
        print("[floodFill]: Done")

if __name__ == '__main__':
    def parse_args():
        """ Parses arguments and returns args object to the main program"""
        parser = argparse.ArgumentParser()
        parser.add_argument('-i', '--inp', type=str, nargs='?',
                            default=DB_PATH,
                            help="Path to the segs h5 dataset.")
        parser.add_argument('-o', '--out', type=str, nargs='?',
                            default=OUT_PATH,
                            help="Path where to save the labeled h5 dataset.")
        parser.add_argument('-v', '--verbose', action="store_true",
                            default=False,
                            help="Print info to the console.")
        return parser.parse_known_args()

    # parse arguments
    ARGS, UNKNOWN = parse_args()

    floodfill_dataset(ARGS.inp, ARGS.out, ARGS.verbose)
