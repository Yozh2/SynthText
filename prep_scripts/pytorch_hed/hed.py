#!/usr/bin/env python

# Standard imports
import os
import os.path as osp
import sys

# Third-party imports
import argparse
import getopt
import h5py
import numpy as np
import PIL
import PIL.Image

# PyTorch imports
import torch
import torch.utils.serialization

##########################################################
assert(int(torch.__version__.replace('.', '')) >= 40) # requires at least pytorch version 0.4.0
torch.set_grad_enabled(False) # make sure to not compute gradients for computational performance
##########################################################

class Network(torch.nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        self.moduleVggOne = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )

        self.moduleVggTwo = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )

        self.moduleVggThr = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )

        self.moduleVggFou = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )

        self.moduleVggFiv = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )

        self.moduleScoreOne = torch.nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.moduleScoreTwo = torch.nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.moduleScoreThr = torch.nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.moduleScoreFou = torch.nn.Conv2d(in_channels=512, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.moduleScoreFiv = torch.nn.Conv2d(in_channels=512, out_channels=1, kernel_size=1, stride=1, padding=0)

        self.moduleCombine = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=5, out_channels=1, kernel_size=1, stride=1, padding=0),
            torch.nn.Sigmoid()
        )

        self.load_state_dict(torch.load('./models/' + arguments_strModel + '.pytorch', map_location='cpu'))
    # end

    def forward(self, tensorInput):
        tensorBlue = (tensorInput[:, 0:1, :, :] * 255.0) - 104.00698793
        tensorGreen = (tensorInput[:, 1:2, :, :] * 255.0) - 116.66876762
        tensorRed = (tensorInput[:, 2:3, :, :] * 255.0) - 122.67891434

        tensorInput = torch.cat([ tensorBlue, tensorGreen, tensorRed ], 1)

        tensorVggOne = self.moduleVggOne(tensorInput)
        tensorVggTwo = self.moduleVggTwo(tensorVggOne)
        tensorVggThr = self.moduleVggThr(tensorVggTwo)
        tensorVggFou = self.moduleVggFou(tensorVggThr)
        tensorVggFiv = self.moduleVggFiv(tensorVggFou)

        tensorScoreOne = self.moduleScoreOne(tensorVggOne)
        tensorScoreTwo = self.moduleScoreTwo(tensorVggTwo)
        tensorScoreThr = self.moduleScoreThr(tensorVggThr)
        tensorScoreFou = self.moduleScoreFou(tensorVggFou)
        tensorScoreFiv = self.moduleScoreFiv(tensorVggFiv)

        tensorScoreOne = torch.nn.functional.interpolate(input=tensorScoreOne, size=(tensorInput.size(2), tensorInput.size(3)), mode='bilinear', align_corners=False)
        tensorScoreTwo = torch.nn.functional.interpolate(input=tensorScoreTwo, size=(tensorInput.size(2), tensorInput.size(3)), mode='bilinear', align_corners=False)
        tensorScoreThr = torch.nn.functional.interpolate(input=tensorScoreThr, size=(tensorInput.size(2), tensorInput.size(3)), mode='bilinear', align_corners=False)
        tensorScoreFou = torch.nn.functional.interpolate(input=tensorScoreFou, size=(tensorInput.size(2), tensorInput.size(3)), mode='bilinear', align_corners=False)
        tensorScoreFiv = torch.nn.functional.interpolate(input=tensorScoreFiv, size=(tensorInput.size(2), tensorInput.size(3)), mode='bilinear', align_corners=False)

        return self.moduleCombine(torch.cat([ tensorScoreOne, tensorScoreTwo, tensorScoreThr, tensorScoreFou, tensorScoreFiv ], 1))
    # end
# end

moduleNetwork = Network().eval()

##########################################################

def estimate(tensorInput):
    tensorOutput = torch.FloatTensor()

    intWidth = tensorInput.size(2)
    intHeight = tensorInput.size(1)

    tensorPreprocessed = tensorInput.view(1, 3, intHeight, intWidth)
    tensorOutput.resize_(1, intHeight, intWidth).copy_(moduleNetwork(tensorPreprocessed)[0, :, :, :])

    tensorInput = tensorInput.cpu()
    tensorOutput = tensorOutput.cpu()
    return tensorOutput

##########################################################

def add_segs_to_db(db_path='segs.h5', segs=None, imgnames=None):
    """
    Add segmentated images and their names
    to the dataset.
    """
    db = h5py.File(db_path,'w')
    db.create_group("ucms")
    names = list()
    ninstance = len(segs)

    for i in range(ninstance):
        dname = "%s_%d"%(imgname, i)

        db['ucms'].create_dataset(dname, data=segs[i])
        names.append(dname)

    # add names
    names_ascii = [n.encode("ascii", "ignore") for n in names]
    db.create_dataset('names', (len(names_ascii),1),'S10', names_ascii)
    db.close()


def process_images(path_input=RAW_DATA_DIR, path_output=OUTPUT_DIR, verbose=False):
    '''Process every raw image in path_input dir and store result in path_output dir'''

    if not osp.exists(path_input):
        if verbose:
            print(f'[HED]: Wrong input path {path_input}')
        raise FileNotFoundError

    # Create torch tensor and load image for every image in the directory
    files = [f for f in os.listdir(path_input) if osp.isfile(osp.join(path_input, f))]
    n = len(files)

    if verbose:
        print(f'[HED]: Found {len(files)} files in {path_input}')

    segs = list()
    for i in range(n):
        img = np.array(PIL.Image.open(path_input))[:, :, ::-1]
        img = img.transpose(2, 0, 1).astype(np.float32) * (1.0 / 255.0)
        tensorInput = torch.FloatTensor(img)
        tensorOutput = estimate(tensorInput)

        # Get segmentation
        seg = (tensorOutput.clamp(0.0, 1.0).np().transpose(1, 2, 0)[:, :, 0] * 255.0).astype(np.uint8)
        segs.append(seg)

    # Save segmentation to the dataset via h5py
    input_dir_name = os.path.basename(RAW_DATA_DIR)
    output_dset_path = osp.join(OUTPUT_DIR, input_dir_name, '.h5')
    add_segs_to_db(segs=segs, db_path=output_dset_path, imgnames=imgnames)


if __name__ == '__main__':
    def parse_args():
        """ Parses arguments and returns args object to the main program"""
        parser = argparse.ArgumentParser()
        parser.add_argument('-m', '--model', type=str, nargs='?',
                            default='bsds500',
                            help="The name of the model to use.")
        parser.add_argument('-i', '--in', type=str, nargs='?',
                            default=osp.join(RAW_DATA_DIR, 'curved_paper.jpg'),
                            help="Path to the imput dir with raw images.")
        parser.add_argument('-o', '--out', type=str, nargs='?',
                            default=osp.join(OUTPUT_DIR, 'curved_paper_depth.jpg'),
                            help="Path to the output dir with depth images and datasets.")
        parser.add_argument('-v', '--verbose', action="store_true",
                            default=False,
                            help="Print info to the console.")
        return parser.parse_known_args()

    # parse arguments
    ARGS, UNKNOWN = parse_args()

    process_images(path_input=ARGS.in, path_output=ARGS.out, verbose=ARGS.VERBOSE)