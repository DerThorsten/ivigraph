from pyqtgraph.flowchart import Flowchart, Node
import pyqtgraph.flowchart.library as fclib
from pyqtgraph.flowchart.library.common import CtrlNode
from pyqtgraph.flowchart import Node
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg
import numpy as np
import scipy.ndimage
import vigra
import math
import functools

from collections import OrderedDict
from nodegraphics import CustomNodeGraphicsItem

from termcolor import colored
import time


# load disk filters

from operators.node_base import convertNh, MyCtrlNode,numpyInNumpyOutNode

import operators.vigra_disk_filters
import operators.vigra_pixel_wise
import operators.vigra_filters
import operators.vigra_recursive_filters
import operators.vigra_tensors
import operators.vigra_analysis
import operators.vigra_sampling
import operators.vigra_machine_learning

import operators.vigra_segmentation #TODO


import operators.numpy_op           #TODO
import operators.channels
import operators.normalize

















##################################################
#
#   SELECTOR
#
###################################################

class Selector(CtrlNode):
    """ Since the windows to show are limited one might need a selector"""
    nodeName = "Selector2"
    uiTemplate=[('selection', 'combo', {'values': ['A', 'B'], 'index': 0})]
    def __init__(self, name):
        terminals = {
            'A': dict(io='in'),
            'B': dict(io='in'),     
            'dataOut': dict(io='out'),  # to specify whether it is input or output
        }                              # other more advanced options are available
                                       # as well..
        CtrlNode.__init__(self, name, terminals=terminals)
    def process(self, A,B, display=True):
        selection=self.ctrls['selection'].currentIndex()
        if selection==0:
            return {'dataOut': A}
        else:
            return {'dataOut': B} 

"""
node = numpyInNumpyOutNode(
    nodeName="selctor2",
    uiTemplate=[
        ('radius','intSpin', {'value': 1, 'min': 1, 'max': 1e9 })
    ],
    f=_normalize
)
"""

fclib.registerNodeType(Selector,[('Data-Selector',)])





###################################################
#
#   Artistic
#
###################################################

def _sepia(data):
    d=data.copy()
    m_sepia = np.asarray([[0.393, 0.769, 0.189],
                             [0.349, 0.686, 0.168],
                             [0.272, 0.534, 0.131]])






    for y in range(data.shape[1]):
        for x in range(data.shape[0]):
            d[x,y,:]=np.dot(d[x,y,:],m_sepia.T)
    return d



node = numpyInNumpyOutNode(
    nodeName="Sepia",
    uiTemplate=[
    ],
    f=_sepia
)
fclib.registerNodeType(node,[('Image-Artistic',)])




###################################################
#
#   WATERSHED
#
###################################################




vigra.analysis.regionImageToCrackEdgeImage


class SegVisu(CtrlNode):
    """ visualization of a segmentaion"""
    nodeName = "SegVisu"

    uiTemplate=[
        #('neighborhood', 'combo', {'values': ['4', '8'], 'index': 0})
    ]

    def __init__(self, name):
        terminals = OrderedDict()
        terminals['labelImage']=dict(io='in')
        terminals['image']=dict(io='in')
        terminals['dataOut']=dict(io='out')
        # as well..
        CtrlNode.__init__(self, name, terminals=terminals)


    def process(self, labelImage,image=None, display=True):
        #nh=4
        #if self.ctrls['neighborhood'].currentIndex() == 1 :
        #    nn=8

        lImg = np.require(labelImage,dtype=np.uint32)

        crackedEdgeImage = vigra.analysis.regionImageToCrackEdgeImage(lImg)

        whereNoEdge = np.where(crackedEdgeImage!=0)
        whereEdge   = np.where(crackedEdgeImage==0)
        crackedEdgeImage[np.where(crackedEdgeImage!=0)]=1


        if image  is not None :
            if image.ndim==3 :
                if tuple(image.shape[0:2]) == tuple(crackedEdgeImage.shape):
                    imgOut=image.copy()
                else:
                    imgOut=vigra.sampling.resize(image,tuple(crackedEdgeImage.shape))

                for c in range(imgOut.shape[2]):
                    imgOut[whereEdge[0],whereEdge[1],c]=0.0

            else :
                if tuple(image.shape[0:2]) == tuple(crackedEdgeImage.shape):
                    imgOut=image.copy()
                else:
                    imgOut=vigra.sampling.resize(image,tuple(crackedEdgeImage.shape))
                imgOut[whereEdge[0],whereEdge[1]]=0.0

            return {'dataOut': imgOut}
        
fclib.registerNodeType(SegVisu ,[('Image-Segmentation',)])







class Watershed(CtrlNode):
    """ (seeded) watershed"""
    nodeName = "Watershed"

    uiTemplate=[('neighborhood', 'combo', {'values': ['4', '8'], 'index': 0})]

    def __init__(self, name):
        terminals = {
            'growImage': dict(io='in'),
            'seedImage': dict(io='in'),     
            'dataOut': dict(io='out')  # to specify whether it is input or output
        }                              # other more advanced options are available
                                       # as well..
        CtrlNode.__init__(self, name, terminals=terminals)
    def process(self, growImage,seedImage=None, display=True):
        nh=4
        if self.ctrls['neighborhood'].currentIndex() == 1 :
            nn=8
        seg,numSeg = vigra.analysis.watersheds(image=growImage,neighborhood=4,seeds=seedImage)
        return {'dataOut': seg}

        
fclib.registerNodeType(Watershed ,[('Image-Segmentation',)])



class SmartWatershed(CtrlNode):
    """ (seeded) watershed"""
    nodeName = "SmartWatershed"

    uiTemplate=[

        ('Seed_NLD_et', 'spin', {'value': 0.25, 'step': 0.05, 'range': [0.01, None]}) ,
        ('Seed_NLD_scale',         'spin', {'value': 1.50, 'step': 0.25, 'range': [0.01, None]}) ,
        ('Seed_GM_sigma',          'spin', {'value': 1.0, 'step': 0.25, 'range': [0.01, None]}) ,
        ('Seed_GM_pow',            'spin', {'value': 2.0, 'step': 0.25, 'range': [0.01, 20]}) ,
        ('Seed_GM_S_sigma',        'spin', {'value': 0.7, 'step': 0.1, 'range': [0.01, 20]}) ,
        ('Seed_LM_neighborhood',   'combo', {'values': ['4', '8'], 'index': 1}),
        ('Seed_LI_neighborhood',   'combo', {'values': ['4', '8'], 'index': 1}),
        ('Grow_NLD_et', 'spin', {'value': 0.25, 'step': 0.05, 'range': [0.01, None]}) ,
        ('Grow_NLD_scale',         'spin', {'value': 1.50, 'step': 0.25, 'range': [0.01, None]}) ,
        ('Grow_GM_sigma',          'spin', {'value': 1.0, 'step': 0.25, 'range': [0.01, None]}) ,
        ('Grow_GM_pow',            'spin', {'value': 2.0, 'step': 0.25, 'range': [0.01, 20]}) ,
        ('Grow_GM_S_sigma',        'spin', {'value': 0.7, 'step': 0.1, 'range': [0.01, 20]}) ,
        ('Grow_neighborhood',      'combo', {'values': ['4', '8'], 'index': 0})
    ]
    def __init__(self, name):
        terminals = {
            'dataIn': dict(io='in'),  
            'dataOut': dict(io='out')  # to specify whether it is input or output
        }                              # other more advanced options are available
                                       # as well..
        CtrlNode.__init__(self, name, terminals=terminals)
    def process(self,dataIn, display=True):

        ################################################
        #   SEEDING
        ###############################################
        # diffuse seeding image
        et    = self.ctrls['Seed_NLD_et'].value()
        scale = self.ctrls['Seed_NLD_scale'].value()
        seedImgDiff = vigra.filters.nonlinearDiffusion(dataIn,edgeThreshold=et,scale=scale)
        
        # compute gradient on seeding image
        sigma  = self.ctrls['Seed_GM_sigma'].value()
        powF   = self.ctrls['Seed_GM_pow'].value()
        sigmaS = self.ctrls['Seed_GM_S_sigma'].value()
        seedGradMag      = vigra.filters.gaussianGradientMagnitude(seedImgDiff,sigma=sigma)
        seedGradMagPow   = seedGradMag**powF
        seedGradSmoothed = vigra.filters.gaussianSmoothing(seedGradMagPow,sigma=sigmaS)

        # get seed image
        nhLM = convertNh(self.ctrls['Seed_LM_neighborhood'].currentIndex())
        nhLI = convertNh(self.ctrls['Seed_LM_neighborhood'].currentIndex())
        local_min=vigra.analysis.localMinima(seedGradSmoothed,neighborhood=nhLM)
        seeds=vigra.analysis.labelImageWithBackground(local_min,neighborhood=nhLI)


        ################################################
        #   evaluation map
        ###############################################
        # diffuse grow image
        et    = self.ctrls['Seed_NLD_et'].value()
        scale = self.ctrls['Seed_NLD_scale'].value()
        # do it
        growImgDiff = vigra.filters.nonlinearDiffusion(dataIn,edgeThreshold=et,scale=scale)
        
        # compute gradient on grow image
        sigma  = self.ctrls['Grow_GM_sigma'].value()
        powF   = self.ctrls['Grow_GM_pow'].value()
        sigmaS = self.ctrls['Grow_GM_S_sigma'].value()
        # do it
        growGradMag      = vigra.filters.gaussianGradientMagnitude(growImgDiff,sigma=sigma)
        growGradMagPow   = growGradMag**powF
        growGradSmoothed = vigra.filters.gaussianSmoothing(growGradMagPow,sigma=sigmaS)

        nh = convertNh(self.ctrls['Grow_neighborhood'].currentIndex())
        # watersheds
        labels,numseg=vigra.analysis.watersheds( 
            image        = growGradSmoothed,
            seeds        = seeds,
            neighborhood = nh,
            method       = 'RegionGrowing'
        )




        #nh=convertNh(self.ctrls['neighborhood'].currentIndex())

        #seg,numSeg = vigra.analysis.watersheds(image=growImage,neighborhood=4,seeds=seedImage)
        return {'dataOut': labels}

        
fclib.registerNodeType(SmartWatershed ,[('Image-Segmentation',)])




def nifty_sp(
    imgRGB,
    edgeThreshold    = 0.25,
    scale            = 20.0,
    sigmaGradMagSeed = 1.5,
    powSeedMap       = 2,
    sigmaSmooth      = 0.7,
    sigmaGradMagGrow = 1.2
):
    assert isinstance(imgRGB, vigra.VigraArray)
    img = vigra.colors.transform_RGB2Lab(imgRGB)
    assert isinstance(img, vigra.VigraArray)
    
    #print "diffuse"
    diffImg = vigra.filters.nonlinearDiffusion(img,edgeThreshold, scale)

    #print "smart watershed"
    # find seeds 
    #print "gaussianGradientMagnitude on diffImg=%r with sigma=%f" % (diffImg.shape, sigmaGradMagSeed)
    seeding_map  = vigra.filters.gaussianGradientMagnitude(diffImg,sigmaGradMagSeed)
    #print "seeding_map: shape=%r" % (seeding_map.shape,)
    seeding_map  = vigra.filters.gaussianSmoothing(seeding_map**powSeedMap,sigmaSmooth)
    local_minima = vigra.analysis.extendedLocalMinima(seeding_map)
    seed_map     = vigra.analysis.labelImageWithBackground(local_minima,neighborhood=8)
    #print "seed_map: %d labels" % seed_map.max()

    # evaluation map
    evaluation_map = vigra.filters.gaussianGradientMagnitude(diffImg,sigmaGradMagGrow)

    # watersheds
    labels,numseg=vigra.analysis.watersheds( 
        image        = evaluation_map,
        seeds        = seed_map,
        neighborhood = 4,
        method       = 'RegionGrowing'
    )


    #print "%d superpixels" % numseg

    #print "get init cgp and resample image"
    #print "numseg",numseg,labels.min(),labels.max()
    cgp,grid=superimg.cgpFromLabels(labels)

    #imgRGBBig = vigra.sampling.resize(img,cgp.shape,0)
    #superimg.visualize(imgRGBBig,cgp,np.ones(cgp.numCells(1),dtype=np.float32), cmap='jet',title='mixed')
   
    assert labels.shape[2] == 1
    labels = labels.squeeze()
    assert labels.ndim == 2, "labels has shape %r" % (labels.shape,)

    return labels,numseg


def _permuteLabels(data):

    flat = np.array(data).reshape([-1])
    unique , relabeling = np.unique(flat,return_inverse=True)
    permUnique = np.random.permutation(unique)
    flatNew = permUnique[relabeling]
    newLabels = flatNew.reshape([data.shape[0],data.shape[1]])
    return newLabels

node = numpyInNumpyOutNode(
    nodeName="PermuteLabels",
    uiTemplate=[
    ],
    f=_permuteLabels
)
fclib.registerNodeType(node,[('Image-Segmentation',)])



###################################################
#
#   Blender
#
###################################################
class Blender(CtrlNode):
    """ blend images (weighted), normalize, if neccessary """
    nodeName = "Blender"

    uiTemplate=[('normalize', 'combo', {'values': ['0', '1', '255']}),
                ('weight', 'spin', {'value' : 0.5, 'step' : 0.05, 'range': [0.0, 1.0]})]

    def __init__(self, name):
        terminals = {
            'Image1': dict(io='in'),
            'Image2': dict(io='in'),
            'BlendedImage': dict(io='out')
        }
        CtrlNode.__init__(self, name, terminals=terminals)
    def process(self, Image1, Image2, display=True):
        if Image1 is None or Image2 is None:
            return {'BlendedImage': np.zeros((0,0))}
        if len(Image1.shape) == 2:
            Image1 = vigra.VigraArray(Image1[..., np.newaxis],
                                      axistags = vigra.VigraArray.defaultAxistags(3))
        if len(Image2.shape) == 2:
            Image2 = vigra.VigraArray(Image2[..., np.newaxis],
                                      axistags = vigra.VigraArray.defaultAxistags(3))
        if Image1.shape[:2] != Image2.shape[:2]:
            raise Exception("Image dimensions disagree!")
        if Image1.shape[2] == 1 and Image2.shape[2] == 3:
            Image1 = np.repeat(Image1, 3, axis=2)
        elif Image1.shape[2] == 3 and Image2.shape[2] == 1:
            Image2 = np.repeat(Image2, 3, axis=2)
        if Image1.shape[2] != Image2.shape[2]:
            raise Exception("Image channels disagree!")

        normalization = int(self.ctrls['normalize'].currentText())
        if normalization > 0:
            Image1 = _normalize(Image1, 0, normalization)
            Image2 = _normalize(Image2, 0, normalization)
        weight = self.ctrls['weight'].value()
        return {'BlendedImage': weight*Image1 + (1-weight)*Image2}
fclib.registerNodeType(Blender, [('Operators',)])




###################################################
#
#   numpy
#
###################################################

_numpyDtypes= [
    np.float32,
    np.float64,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.bool
]

_numpyDtypeStrs = [ str(d) for d in _numpyDtypes]


_stringedBooleanOperators = {
    '=='    : '__eq__' , 
    '!='    : '__ne__' , 
    '<'     : '__lt__' , 
    '>'     : '__gt__' , 
    '<='    : '__le__' , 
    '>='    : '__ge__' 
}

class NumpyRequire(CtrlNode):
    """ blend images (weighted), normalize, if neccessary """
    nodeName = "numpy.require"
    uiTemplate = [
        ('dtype', 'combo', {'values': _numpyDtypeStrs})
    ]
    def __init__(self, name):
        terminals = {
            'dataIn': dict(io='in'),
            'dataOut': dict(io='out')
        }
        CtrlNode.__init__(self, name, terminals=terminals)
    def process(self, dataIn, display=True):

        return { 'dataOut': 
            np.require(dataIn,dtype=_numpyDtypes[  self.ctrls['dtype'].currentIndex()   ])
        }

fclib.registerNodeType(NumpyRequire, [('Numpy',)])





class NumpyWhereNot(CtrlNode):
    """ blend images (weighted), normalize, if neccessary """
    nodeName = "numpy.whereNot"

    uiTemplate=[('isNotValue', 'spin', {'value' : 0, 'step' : 1, 'range': [0, None]})]

    def __init__(self, name):
        terminals = {
            'Image': dict(io='in'),
            'Indices': dict(io='out')
        }
        CtrlNode.__init__(self, name, terminals=terminals)
    def process(self, Image, display=True):
        ignore_value = self.ctrls['ignore'].value()
        if ignore_value is not None:
            return {'Indices': np.where(Image != ignore_value)}
        else:
            return {'Indices': None}
fclib.registerNodeType(NumpyWhereNot, [('Numpy',)])
