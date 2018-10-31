"""
Python script to "flood-fill" the segments computed using gPb-UCM.
This assings the same integer label to all the pixels in the same segment.

Author: Ankush Gupta
"""

from __future__ import division
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import cv2
import scipy.io as sio
import h5py
import os.path as osp
import multiprocessing as mp
import traceback, sys

def get_seed(sx,sy,ucm):
    n = sx.size
    for i in range(n):
        if ucm[sx[i]+1,sy[i]+1] == 0:
            return (sy[i],sx[i])

def get_mask(ucm,viz=False):
    ucm = ucm.copy()
    h,w = ucm.shape[:2]
    mask = np.zeros((h-2,w-2),'float32')

    i = 0
    sx,sy = np.where(mask==0)
    seed = get_seed(sx,sy,ucm)
    areas = []
    labels=[]
    while seed is not None and i<1000:
        cv2.floodFill(mask,ucm,seed,i+1)
        # calculate the area (no. of pixels):
        areas.append(np.sum(mask==i+1))
        labels.append(i+1)

        # get the location of the next seed:
        sx,sy = np.where(mask==0)
        seed = get_seed(sx,sy,ucm)
        i += 1
    print("  > terminated in %d steps" % i)

    if viz:
        plt.imshow(mask)
        plt.show()

    return mask,np.array(areas),np.array(labels)

def get_mask_parallel(ucm_imname):
    ucm,imname = ucm_imname
    try:
        return (get_mask(ucm.T),imname)
    except:
        traceback.print_exc(file=sys.stdout)
        return None

def process_db_parallel(base_dir, db_path, dbo_mask, th=0.11):
    """
    Get segmentation masks from gPb contours.
    """
    class ucm_iterable(object):
        def __init__(self,ucm_path,th):
            self.th = th
            self.ucm_h5 = h5py.File(db_path,'r')
            self.N = self.ucm_h5['names'].size
            self.i = 0

        def __iter__(self):
            return self

        def get_imname(self,i):
            return "".join(map(chr, self.ucm_h5['names'][0,self.i][:]))

        def __stop__(self):
            print("DONE")
            self.ucm_h5.close()
            raise StopIteration

        def get_valid_name(self):
            if self.i >= self.N:
                self.__stop__()

            imname = self.get_imname(self.i)
            while self.i < self.N-1 and len(imname) < 4:
                self.i += 1
                imname = self.get_imname(self.i)

            if len(imname) < 4:
                self.__stop__()

            return imname

        def __next__(self):
            imname = self.get_valid_name()
            print("%d of %d" % (self.i + 1, self.N))
            keys = list(self.ucm_h5['ucms'].keys())
            ucm = self.ucm_h5['ucms'][keys[self.i]][:]
            ucm = ucm.copy()
            self.i += 1
            return ((ucm > self.th).astype(np.uint8), imname)

    ucm_iter = ucm_iterable(db_path, th)
    cpu_count = mp.cpu_count()
    print("cpu count: ", cpu_count)
    parpool = mp.Pool(cpu_count)
    
#     ucm_result = list()
#     for ucm in ucm_iter:
#         ucm_result.append(get_mask_parallel
    
    ucm_result = parpool.imap_unordered(get_mask_parallel, ucm_iter, chunksize=1)
    print(ucm_result.__class__)

    for res in ucm_result:
        if res is None:
            continue
        ((mask,area,label), imname) = res
        print("got back : ", imname)
        mask = mask.astype('uint16')
        mask_dset = dbo_mask.create_dataset(imname, data=mask.T)
        mask_dset.attrs['area'] = area
        mask_dset.attrs['label'] = label

# Setup paths
base_dir = '/home/gayduchenko/data/' # directory containing the ucm.mat, i.e., output of run_ucm.m
db_path = osp.join(base_dir,'curved_paper_segmented.jpg.h5')
out_path = osp.join(base_dir,'curved_paper_labels.h5')

# output h5 file:
print('Creating output h5 file %s' % out_path)
dbo = h5py.File(out_path,'w')
dbo_mask = dbo.create_group("mask")    
    
process_db_parallel(base_dir, db_path, dbo_mask)

# close the h5 files:
print("closing DB")
dbo.close()
print(">>>> DONE")