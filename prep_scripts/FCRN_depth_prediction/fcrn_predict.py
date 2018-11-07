# System imports
import os
import os.path as osp

# Third-party imports
import argparse
import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tqdm import tqdm

# Local imports
import models

# Define default constants
MY_DIR = osp.dirname(osp.abspath(__file__))
FCRN_MODEL_PATH = osp.join(MY_DIR, 'models/NYU_FCRN.ckpt')
FCRN_INPUT_PATH = osp.join(MY_DIR, '../../data/images/raw')
FCRN_OUTPUT_PATH = osp.join(MY_DIR, '../../data/images/depths')
FCRN_IMAGE_SHAPE = (1280*2, 720*2) # height, width in pixels
VERBOSE = False

class FCRNDepthPredictor:
    def __init__(self, path_model=FCRN_MODEL_PATH,
                 image_shape=FCRN_IMAGE_SHAPE, channels=3, batch_size=1, verbose=False):
        '''Construct TensorFlow graph to predict depth in the future.

        Args:
            path_model (str):
                path to pretrained model weights

            path_input (str):
                path to the directory with input raw RGB images

            path_output (str):
                path to the directory the depth dataset will be stored into

            channels (int):
                number of image channels, such as RGB (channels == 3)

            batch_size (int):
                size of the batch size for ResNet

            verbose (bool):
                toggle print debug information to the console
        '''
        self.verbose = verbose
        self.path_model = path_model

        # Remember tensorflow session
        self.batch_size = batch_size
        self.channels = channels
        self.saver = None

        # Create a placeholder for the input image
        self.input_node = tf.placeholder(tf.float32, shape=(None, image_shape[0], image_shape[1], channels))
        if self.verbose:
            print('[FCRN]: Input tensorflow placeholder created...')

        # Construct the network
        self.net = models.ResNet50UpProj({'data': self.input_node},
                                         self.batch_size, 1, False)
        if self.verbose:
            print('[FCRN]: ResNet50 initialized...')


    def load_weights(self, sess=None):
        '''Load model's pretrained weights'''
        # Use to load from ckpt file
        self.saver = tf.train.Saver()
        self.saver.restore(sess, self.path_model)

        # Use to load from npy file
        #net.load(model_data_path, sess)

    def load_image(self, path_input, shape=FCRN_IMAGE_SHAPE):
        '''Load and preprocess image from `path_input` given.'''
        # Load original image
        if not os.path.exists(path_input):
            if self.verbose:
                print(f'[FCRN]: Image {path_input} was not found!')
            raise FileNotFoundError
        else:
            image_cv2 = cv2.imread(path_input)
        # if self.verbose:
        #     print(f'[FCRN]: Original image loaded with shape: {image_cv2.shape}')

        # Resize image to the requested shapes
        height, width = shape[0], shape[1]
        if height is None:
            height = image_cv2.shape[0]
        if width is None:
            width = image_cv2.shape[1]
        image_cv2 = cv2.resize(image_cv2, (width, height),
            interpolation = cv2.INTER_AREA)
        # if self.verbose:
        #     print(f'[FCRN]: Resized image shape: {image_cv2.shape}')

        # Preprocess image to use it with tensorflow
        image_cv2 = image_cv2.astype('float32')
        image_cv2 = np.expand_dims(image_cv2, axis=0)
        # if self.verbose:
        #     print(f'[FCRN]: Postprocessed image shape: {image_cv2.shape}')
        return image_cv2

    # def save_depth(self, depth, path):
    #     '''DEPRICATED: Save depth as numpy pickle and as image'''
    #     fig = plt.figure()
    #     ii = plt.imshow(depth[0, :, :, 0], interpolation='nearest')
    #     fig.colorbar(ii)
    #     plt.savefig('%s.png' % path)
    #     np.save('%s.npy' % path, depth)

    def predict_dir(self, path_input=FCRN_INPUT_PATH, path_output=FCRN_OUTPUT_PATH, sess=None):
        '''Predict depth for all images in the path_input'''
        if not os.path.exists(path_input):
            if self.verbose:
                print(f'[FCRN]: Image input directory {path_input} was not found!')
            raise NotADirectoryError

        # load images
        files = [f for f in os.listdir(path_input) if osp.isfile(osp.join(path_input, f))]
        n = len(files)
        if self.verbose:
            print(f'[FCRN]: Found {n} files in {path_input}')

        if not osp.exists(path_output):
            if self.verbose:
                print(f'[FCRN]: Creating output directory {path_output} as it was not found.')
            os.makedirs(path_output)

        # create and setup h5 dataset file
        dataset_name = 'depths.h5'
        dataset_path = osp.abspath(osp.join(path_output, dataset_name))
        depths = h5py.File(dataset_path, 'w')
        depths.create_group('depths')

        names_ascii = [n.encode("ascii", "ignore") for n in files]
        depths.create_dataset('names', (len(names_ascii),1),'S256', names_ascii)

        for i in tqdm(range(n), desc='[FCRN]: Processing images'):
            # load image
            path_image = os.path.join(path_input, files[i])
            image = self.load_image(path_image)

            # estimate depth
            pred = self.predict(image, sess)

            # store depth into dataset
            # print(f'Saving data {i+1} of {n}')
            depths['depths'].create_dataset(files[i], data=pred)

        depths.close()
        return path_output

    def predict(self, image_cv2, sess):
        '''Evalute the network for the given image'''
        pred = sess.run(self.net.get_output(), feed_dict={self.input_node: image_cv2})
        return pred

def predict_depth(path_model=FCRN_MODEL_PATH, path_input=FCRN_INPUT_PATH,
                  path_output=FCRN_OUTPUT_PATH, verbose=False):

    fcrn = FCRNDepthPredictor(path_model, FCRN_IMAGE_SHAPE,
                              channels=3, batch_size=1,
                              verbose=verbose)

    with tf.Session() as sess:
        fcrn.load_weights(sess)

        if args.verbose:
            print('='*50 +'\n\n')

        preds_path = fcrn.predict_dir(path_input, path_output, sess)
        if verbose:
            print(f'[FCRN]: Estimated depths are stored in {preds_path}')
            print('[FCRN]: Done.')

if __name__ == '__main__':
    def parse_args():
        """Parses arguments and returns args object to the main program"""
        parser = argparse.ArgumentParser()
        parser.add_argument('-m', '--model', default=FCRN_MODEL_PATH,
                            help='Converted parameters for the model')
        parser.add_argument('-i', '--inp', default=FCRN_INPUT_PATH,
                            help='Path to directory with input images')
        parser.add_argument('-o', '--out', default=FCRN_OUTPUT_PATH,
                            help='Path to the directory the output dataset will be stored in')
        parser.add_argument('-v', '--verbose', action='store_true', default=VERBOSE,
                            help='Print debug information to the console')
        return parser.parse_known_args()

    # parse arguments
    args, unknown = parse_args()

    predict_depth(args.model, args.inp, args.out, args.verbose)
