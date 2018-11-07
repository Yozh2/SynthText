# Author: Ankush Gupta
# Date: 2015

"""
Entry-point for generating synthetic text images, as described in:

@InProceedings{Gupta16,
      author       = "Gupta, A. and Vedaldi, A. and Zisserman, A.",
      title        = "Synthetic Data for Text Localisation in Natural Images",
      booktitle    = "IEEE Conference on Computer Vision and Pattern Recognition",
      year         = "2016",
    }
"""
import numpy as np
import h5py
import os, sys, traceback
import os.path as osp
from synthgen import *
from common import *
import wget, tarfile

## Define some configuration variables:
NUM_IMG = -1 # no. of images to use for generation (-1 to use all available):
INSTANCE_PER_IMAGE = 1 # no. of times to use the same image
SECS_PER_IMG = 5 # max time per image in seconds

# path to the data-file, containing image, depth and segmentation:
DATA_PATH = './data'
TXTDATA_PATH = osp.join(DATA_PATH, 'newsgroup', 'newsgroup.txt')
IMAGES_PATH = osp.join(DATA_PATH, 'images')
DB_FNAME = osp.join(IMAGES_PATH,'dset.h5')
# url of the data (google-drive public file):
DATA_URL = 'http://www.robots.ox.ac.uk/~ankush/data.tar.gz'
OUT_FILE ='results/SynthText.h5'

def get_data(db_fname=DB_FNAME, data_url=DATA_URL):
  """
  Download the image,depth and segmentation data:
  Returns, the h5 database.
  """
  if not osp.exists(DB_FNAME):
    try:
      colorprint(Color.RED, '[SynthText]: No dataset found')
      colorprint(Color.YELLOW, '[SynthText]: Do you want to download demo?')
      ans = input('y/[N]').lower()
      if ans in ['no', 'n', '']:
        exit(0)
      elif ans not in ['yes', 'y']:
        colorprint(Color.YELLOW, '[SynthText]: Wrong answer {answer}')
        exit(0)

      colorprint(Color.BLUE,'[SynthText]: downloading data (56 M) from: '+DATA_URL,bold=True)
      print()
      sys.stdout.flush()
      out_fname = 'data.tar.gz'
      wget.download(DATA_URL,out=out_fname)
      tar = tarfile.open(out_fname)
      tar.extractall()
      tar.close()
      os.remove(out_fname)
      colorprint(Color.BLUE,'\n\t[SynthText]: data saved at:' + db_fname, bold=True)
      sys.stdout.flush()
    except:
      print (colorize(Color.RED,'[SynthText]: Data not found and have problems downloading.',bold=True))
      sys.stdout.flush()
      sys.exit(-1)

  # open the h5 file and return:
  db = h5py.File(db_fname, 'r')
  print('[SynthText]: Opened dataset file', db_fname)
  return db

def add_res_to_db(imgname,res,db):
  """
  Add the synthetically generated text image instance
  and other metadata to the dataset.
  """
  ninstance = len(res)
  for i in range(ninstance):
    dname = "%s_%d"%(imgname, i)
    db['data'].create_dataset(dname,data=res[i]['img'])
    db['data'][dname].attrs['charBB'] = res[i]['charBB']
    db['data'][dname].attrs['wordBB'] = res[i]['wordBB']
    #db['data'][dname].attrs['txt'] = res[i]['txt']
    L = res[i]['txt']
    L = [n.encode("ascii", "ignore") for n in L]
    db['data'][dname].attrs['txt'] = L


def main(in_db_path=DB_FNAME, out_db_path=OUT_FILE, data_path=DATA_PATH, txtdata_path=TXTDATA_PATH, viz=False):
  # open databases:
  print (colorize(Color.BLUE,'[SynthText]: getting data..',bold=True))
  db = get_data(in_db_path)
  print (colorize(Color.BLUE,'\t[SynthText]: Got data',bold=True))

  # open the output h5 file:
  if not osp.exists(out_db_path):
    os.makedirs(osp.dirname(osp.abspath(out_db_path)), exist_ok=True)
  out_db = h5py.File(osp.abspath(out_db_path), 'w')
  out_db.create_group('/data')
  print (colorize(Color.GREEN,'[SynthText]: Storing the output in: '+out_db_path, bold=True))

  # get the names of the image files in the dataset:
  imnames = sorted(db['image'].keys())
  N = len(imnames)
  global NUM_IMG
  if NUM_IMG < 0:
    NUM_IMG = N
  start_idx,end_idx = 0,min(NUM_IMG, N)

  colorprint(Color.YELLOW, '[SynthText]: Initializing RV3')
  RV3 = RendererV3(data_path, txtdata_path, max_time=SECS_PER_IMG)
  colorprint(Color.YELLOW, '[SynthText]: Initializing RV3: DONE')
  for i in range(start_idx,end_idx):
    imname = imnames[i]
    try:
      # get the image:
      img_raw = db['image'][imname][:]
      # img_raw[img_raw < 1] = 1
      img = Image.fromarray(img_raw)
      # get the pre-computed depth:
      #  there are 2 estimates of depth (represented as 2 "channels")
      #  here we are using the second one (in some cases it might be
      #  useful to use the other one):
      # depth = db['depth'][imname][:].T
      # depth = depth[:,:,1]
      depth = db['depth'][imname][:]

      # get segmentation:
      seg = db['seg'][imname][:].astype('float32')
      area = db['seg'][imname].attrs['area']
      label = db['seg'][imname].attrs['label']

      # re-size uniformly:
      sz = depth.shape[:2][::-1]
      img = np.array(img.resize(sz,Image.ANTIALIAS))
      seg = np.array(Image.fromarray(seg).resize(sz,Image.NEAREST))

      print (colorize(Color.RED, f'{i+1} of {end_idx}', bold=True))
      res = RV3.render_text(img,depth,seg,area,label,
                            ninstance=INSTANCE_PER_IMAGE,viz=viz)
      if len(res) > 0:
        # non-empty : successful in placing text:
        add_res_to_db(imname,res,out_db)
      # visualize the output:
      if viz:
        if 'q' in input(colorize(Color.RED,'[SynthText]: continue? (enter to continue, q to exit): ',True)):
          break
    except:
      traceback.print_exc()
      print (colorize(Color.GREEN,'[SynthText]: >>>> CONTINUING....', bold=True))
      continue
  db.close()
  out_db.close()


if __name__=='__main__':
  import argparse
  def parse_args():
    """ Parses arguments and returns args object to the main program"""
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--inp', type=str, nargs='?',
                        default=DB_FNAME,
                        help="Path to the input h5 dataset.")
    parser.add_argument('-o', '--out', type=str, nargs='?',
                        default=OUT_FILE,
                        help="Path where to save the output h5 dataset.")
    parser.add_argument('-d', '--data', type=str, nargs='?',
                        default=DATA_PATH,
                        help="The directory where all the data is stored.")
    parser.add_argument('-t', '--txtdata', type=str, nargs='?',
                        default=TXTDATA_PATH,
                        help="Path to the .txt file where all the text data is stored.")
    parser.add_argument('--viz', action='store_true', dest='viz', default=False,
                        help='flag for turning on visualizations')
    return parser.parse_known_args()

  # parse arguments
  ARGS, UNKNOWN = parse_args()
  main(in_db_path=ARGS.inp, out_db_path=ARGS.out, data_path=ARGS.data, viz=ARGS.viz)
