import numpy as np
import vigra
import math
from collections import OrderedDict

from node_base import convertNh, MyNode,MyCtrlNode,numpyInNumpyOutNode
import pyqtgraph.flowchart.library as fclib






###################################################
#
#   Analysis
#
###################################################



def _kammlinie(img,extrema):

    resX=np.zeros(img.shape)
    resY=np.zeros(img.shape)
    if(extrema==0):
        for x in range(img.shape[0]):
            if(x>0 and x<img.shape[0]-1):

                A = img[x-1,:]
                B = img[x  ,:]
                C = img[x+1,:]
                R1 = np.array(B < A).astype(np.float32)
                R2 = np.array(B < C).astype(np.float32)
                resX[x,:]=R1*R2
        for y in range(img.shape[1]):
            if(y>0 and y<img.shape[1]-1):
                A = img[:,y-1]
                B = img[:,y  ]
                C = img[:,y+1]
                R1 = np.array(B < A).astype(np.float32)
                R2 = np.array(B < C).astype(np.float32)
                resY[:,y]=R1*R2

    else :
        for x in range(img.shape[0]):
            if(x>0 and x<img.shape[0]-1):

                A = img[x-1,:]
                B = img[x  ,:]
                C = img[x+1,:]
                R1 = np.array(B > A).astype(np.float32)
                R2 = np.array(B > C).astype(np.float32)
                resX[x,:]=R1*R2
        for y in range(img.shape[1]):
            if(y>0 and y<img.shape[1]-1):
                A = img[:,y-1]
                B = img[:,y  ]
                C = img[:,y+1]
                R1 = np.array(B > A).astype(np.float32)
                R2 = np.array(B > C).astype(np.float32)
                resY[:,y]=R1*R2
    res = resX+resY 
    res[np.where(res>1)]=1.0
    return res*2.0



def _slicSuperpixelsImageOnly(image, **kwargs):
    return vigra.analysis.slicSuperpixels(image, **kwargs)[0]


node = numpyInNumpyOutNode(
    nodeName="SlicSuperpixels",
    uiTemplate=[
        ('intensityScaling', 'spin', {'value': 6.0, 'step': 0.5, 'range': [0.0, None]}),
        ('seedDistance', 'intSpin', {'value': 15, 'step': 1, 'range': [1, None]}),
        ('minSize', 'intSpin', {'value': 0, 'step': 1, 'range': [0, None]}),
        ('iterations', 'intSpin', {'value': 10, 'step': 1, 'range': [1, None]})
    ],
    f=_slicSuperpixelsImageOnly,
    doChannelWise=False
)
fclib.registerNodeType(node,[('Image-Analysis',)])


node = numpyInNumpyOutNode(
    nodeName="ExtremaLines",
    uiTemplate=[
        ('extrema', 'combo', {'values': ['min', 'max'], 'index': 0})
    ],
    f=_kammlinie,
    doChannelWise=True,
    dtypeOut=np.uint32,
)
fclib.registerNodeType(node,[('Image-Analysis',)])


def _localExtrema(img,extrema,marker,neighborhood,extendedExtrema):
    neighborhood=convertNh(neighborhood)
    if(extrema==0):
        if extendedExtrema==False:
            return vigra.analysis.localMinima(img,marker=marker,neighborhood=neighborhood)
        else:
            return vigra.analysis.extendedLocalMinima(img,marker=marker,neighborhood=neighborhood)
    elif(extrema==1):
        if extendedExtrema==False:
            return vigra.analysis.localMaxima(img,marker=marker,neighborhood=neighborhood)
        else:
            return vigra.analysis.extendedLocalMaxima(img,marker=marker,neighborhood=neighborhood)
    else:
        assert False

def _labelImage(img,neighborhood,withBackground,backgroundValue):
    neighborhood=convertNh(neighborhood)
    if(withBackground):
        return vigra.analysis.labelImageWithBackground( img,neighborhood=neighborhood,
                                                        background_value=backgroundValue)
    else:
        return vigra.analysis.labelImage(img,neighborhood=neighborhood)



node = numpyInNumpyOutNode(
    nodeName="LocalExtrema",
    uiTemplate=[
        ('extrema', 'combo', {'values': ['min', 'max'], 'index': 0}),
        ('marker','intSpin', {'value': 2, 'min': 0, 'max': 1e9 }),
        ('neighborhood', 'combo', {'values': ['4', '8'], 'index': 0}),
        ('extendedExtrema', 'check',   {'value': False})
    ],
    f=_localExtrema,
    doChannelWise=True,
    dtypeOut=np.uint32,
)
fclib.registerNodeType(node,[('Image-Analysis',)])



node = numpyInNumpyOutNode(
    nodeName="LabelImage",
    uiTemplate=[
        ('neighborhood', 'combo', {'values': ['4', '8'], 'index': 0}),
        ('withBackground', 'check',   {'value': False}),
        ('backgroundValue','intSpin', {'value': 2, 'min': 0, 'max': 1e9 }),
    ],
    f=_labelImage,
    doChannelWise=True,
    dtypeIn=np.float32,
    dtypeOut=np.uint32
)
fclib.registerNodeType(node,[('Image-Analysis',)])


node = numpyInNumpyOutNode(
    nodeName="CannyEdgeImage",
    uiTemplate=[
       ('scale',     'spin',    {'value': 1.0, 'step': 0.5, 'range': [0.01, None]}),
       ('threshold', 'spin',    {'value': 0.5, 'step': 0.5, 'range': [0.01, None]}),
       ('edgeMarker','intSpin', {'value': 1, 'min': 0, 'max': 1e9 })
    ],
    f=vigra.analysis.cannyEdgeImage,
    doChannelWise=True
)
fclib.registerNodeType(node,[('Image-Analysis',)])


node = numpyInNumpyOutNode(
    nodeName="CannyEdgeImageWithThinning",
    uiTemplate=[
       ('scale',     'spin',    {'value': 1.0, 'step': 0.5, 'range': [0.01, None]}),
       ('threshold', 'spin',    {'value': 0.5, 'step': 0.5, 'range': [0.01, None]}),
       ('edgeMarker','intSpin', {'value': 1, 'min': 0, 'max': 1e9 }),
       ('addBorder', 'check',   {'value': True})
    ],
    f=vigra.analysis.cannyEdgeImageWithThinning,
    doChannelWise=True
)

fclib.registerNodeType(node,[('Image-Analysis',)])


node = numpyInNumpyOutNode(
    nodeName="CornernessBeaudet",
    uiTemplate=[
       ('scale',     'spin',    {'value': 1.0, 'step': 0.5, 'range': [0.01, None]})
    ],
    f=vigra.analysis.cornernessBeaudet,
    doChannelWise=True
)
fclib.registerNodeType(node,[('Image-Analysis',)])



node = numpyInNumpyOutNode(
    nodeName="CornernessBoundaryTensor",
    uiTemplate=[
       ('scale',     'spin',    {'value': 1.0, 'step': 0.5, 'range': [0.01, None]})
    ],
    f=vigra.analysis.cornernessBoundaryTensor,
    doChannelWise=True
)
fclib.registerNodeType(node,[('Image-Analysis',)])


node = numpyInNumpyOutNode(
    nodeName="CornernessFoerstner",
    uiTemplate=[
       ('scale',     'spin',    {'value': 1.0, 'step': 0.5, 'range': [0.01, None]})
    ],
    f=vigra.analysis.cornernessFoerstner,
    doChannelWise=True
)
fclib.registerNodeType(node,[('Image-Analysis',)])


node = numpyInNumpyOutNode(
    nodeName="CornernessHarris",
    uiTemplate=[
       ('scale',     'spin',    {'value': 1.0, 'step': 0.5, 'range': [0.01, None]})
    ],
    f=vigra.analysis.cornernessHarris,
    doChannelWise=True
)
fclib.registerNodeType(node,[('Image-Analysis',)])


node = numpyInNumpyOutNode(
    nodeName="CornernessRohr",
    uiTemplate=[
       ('scale',     'spin',    {'value': 1.0, 'step': 0.5, 'range': [0.01, None]})
    ],
    f=vigra.analysis.cornernessRohr,
    doChannelWise=True
)
fclib.registerNodeType(node,[('Image-Analysis',)])



node = numpyInNumpyOutNode(
    nodeName="RegionToEdges",
    uiTemplate=[
        ('edgeLabel', 'intSpin', {'value': 1, 'step': 1, 'range': [0, None]})
    ],
    f=vigra.analysis.regionImageToEdgeImage,
    dtypeIn=np.uint32,
    doChannelWise=False
)
fclib.registerNodeType(node,[('Image-Analysis',)])