#!/usr/bin/env python

import getopt
import math
import numpy
import os
import PIL
import PIL.Image
import sys
import torch
import torch.utils.serialization
# Custom imports
import numpy as np
import h5py

##########################################################

assert(int(torch.__version__.replace('.', '')) >= 40) # requires at least pytorch version 0.4.0

torch.set_grad_enabled(False) # make sure to not compute gradients for computational performance

##########################################################

arguments_strModel = 'bsds500'
arguments_strIn = './images/sample.png'
arguments_strOut = './out.png'

for strOption, strArgument in getopt.getopt(sys.argv[1:], '', [ strParameter[2:] + '=' for strParameter in sys.argv[1::2] ])[0]:
    if strOption == '--model':
        arguments_strModel = strArgument # which model to use

    elif strOption == '--in':
        arguments_strIn = strArgument # path to the input image

    elif strOption == '--out':
        arguments_strOut = strArgument # path to where the output should be stored

    # end
# end

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

    #	assert(intWidth == 480) # remember that there is no guarantee for correctness, comment this line out if you acknowledge this and want to continue
    #	assert(intHeight == 320) # remember that there is no guarantee for correctness, comment this line out if you acknowledge this and want to continue

    if True:
        tensorInput = tensorInput
        tensorOutput = tensorOutput
    # end

    if True:
        tensorPreprocessed = tensorInput.view(1, 3, intHeight, intWidth)

        tensorOutput.resize_(1, intHeight, intWidth).copy_(moduleNetwork(tensorPreprocessed)[0, :, :, :])
    # end

    if True:
        tensorInput = tensorInput.cpu()
        tensorOutput = tensorOutput.cpu()
    # end

    return tensorOutput
# end

##########################################################

def add_segs_to_db(db_path='segs.h5', segs=None, imgname='image'):
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

if __name__ == '__main__':
    tensorInput = torch.FloatTensor(numpy.array(PIL.Image.open(arguments_strIn))[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32) * (1.0 / 255.0))

    tensorOutput = estimate(tensorInput)

    segs = list()

    # Get segmentation
    seg = (tensorOutput.clamp(0.0, 1.0).numpy().transpose(1, 2, 0)[:, :, 0] * 255.0).astype(numpy.uint8)
    segs.append(seg)
    
    # Save image to the filesystem
    PIL.Image.fromarray(seg).save(arguments_strOut)
    
    # Save segmentation to the dataset via h5py
    add_segs_to_db(segs=segs, db_path=arguments_strOut+'.h5')
    
# end
