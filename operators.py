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
from scipy.stats.mstats import mquantiles
from collections import OrderedDict
from nodegraphics import CustomNodeGraphicsItem

def convertNh(nh):
    if nh == 0 : neighborhood = 4
    else: neighborhood =  8
    return neighborhood


def vigraNode(nodeName,uiTemplate,f,dtypeIn=np.float32,dtypeOut=np.float32,doChannelWise=False):
    name = nodeName
    uiT  = uiTemplate
    kwargs = {}
    for uit in uiTemplate:
        kwargs[uit[0]]=None



    class _VigraNodeImpl(CtrlNode):
        """%s"""%name
        nodeName = name
        uiTemplate = uiT
        def __init__(self, name):
            ## Define the input / output terminals available on this node
            terminals = {
                'dataIn': dict(io='in'),    # each terminal needs at least a name and
                'dataOut': dict(io='out'),  # to specify whether it is input or output
            }                              # other more advanced options are available
                                           # as well..
            
            CtrlNode.__init__(self, name, terminals=terminals)
            
        def process(self, dataIn, display=True):
            if dataIn is None :
                assert False
            # CtrlNode has created self.ctrls, which is a dict containing {ctrlName: widget}
            print "process ",_VigraNodeImpl.nodeName
            for uit in uiTemplate:
                pName = uit[0]
                pType = uit[1]

                print "name ",pName ,"pType",pType
                if(pType=='spin' or pType == 'intSpin'):
                    kwargs[pName]=self.ctrls[pName].value()
                elif(pType=='check'):
                    kwargs[pName]=self.ctrls[pName].isChecked()
                elif(pType=='combo'):
                    kwargs[pName]=self.ctrls[pName].currentIndex()


            #sigma = self.ctrls['sigma'].value()
            dataInVigra = np.require(dataIn,dtype=dtypeIn)
            dataInVigra = np.squeeze(dataInVigra)
            

            if doChannelWise == False or dataInVigra.ndim==2 or dataInVigra.shape[2]==1:
                print "Single Input  ",dataInVigra.shape,dataInVigra.dtype
                vigraResult = f(dataInVigra,**kwargs)
            else:

                numChannels = dataInVigra.shape[2]
                vigraResult  = np.ones( dataInVigra.shape,dtype=dtypeIn)
                for c in range(numChannels):
                    print "channel wise input :",dataInVigra[:,:,c].shape,dataInVigra[:,:,c].dtype
                    vigraResult[:,:,c]=f(dataInVigra[:,:,c],**kwargs)
            vigraResult = np.squeeze(vigraResult)
            vigraResult = np.require(vigraResult,dtype=dtypeOut)


            return {'dataOut': vigraResult}

    return _VigraNodeImpl        










#######################################################
# 
#   PIXEL-WISE
#
########################################################


def _brightness(image,factor):
    return vigra.colors.brightness(image,factor=float(factor),range='auto')
def _contrast(image,factor):
    return vigra.colors.contrast(image,factor=float(factor),range='auto')
def _gammaCorrection(image,gamma):
    return vigra.colors.gammaCorrection(image,gamma=float(gamma),range='auto')

node = vigraNode(
    nodeName="Brightness",
    uiTemplate=[  ('factor',  'spin', {'value': 1.00, 'step': 0.20, 'range': [0.10, None]}) ],
    f=_brightness,
)
fclib.registerNodeType(node,[('Image-Color/Intensity',)])

node = vigraNode(
    nodeName="Contrast",
    uiTemplate=[  ('factor',  'spin', {'value': 1.00, 'step': 0.20, 'range': [0.10, None]}) ],
    f=_contrast,
)
fclib.registerNodeType(node,[('Image-Color/Intensity',)])

node = vigraNode(
    nodeName="GammaCorrection",
    uiTemplate=[  ('gamma',  'spin', {'value': 1.00, 'step': 0.20, 'range': [0.10, None]}) ],
    f=_gammaCorrection,
)
fclib.registerNodeType(node,[('Image-Color/Intensity',)])



#######################################################
# 
#   ColorTransform
#
########################################################
def addColorTransform(f,name):
    node = vigraNode(
        nodeName=name,
        uiTemplate=[],
        f=f,
    )
    fclib.registerNodeType(node,[('Image-ColorSpace',)])

addColorTransform( f=vigra.colors.transform_Lab2RGB , name ="Lab2RGB")
addColorTransform( f=vigra.colors.transform_Lab2RGBPrime , name ="Lab2RGBPrime")
addColorTransform( f=vigra.colors.transform_Lab2XYZ , name ="Lab2XYZ")
addColorTransform( f=vigra.colors.transform_Luv2RGB , name ="Luv2RGB")
addColorTransform( f=vigra.colors.transform_Luv2RGBPrime , name ="Luv2RGBPrime" )
addColorTransform( f=vigra.colors.transform_Luv2XYZ , name ="Luv2XYZ")
addColorTransform( f=vigra.colors.transform_RGB2Lab , name ="RGB2Lab")
addColorTransform( f=vigra.colors.transform_RGB2Luv , name ="RGB2Luv")
addColorTransform( f=vigra.colors.transform_RGB2RGBPrime , name ="RGB2RGBPrime")
addColorTransform( f=vigra.colors.transform_RGB2sRGB , name ="RGB2sRGB")
addColorTransform( f=vigra.colors.transform_RGBPrime2Lab , name ="RGBPrime2Lab")
addColorTransform( f=vigra.colors.transform_RGBPrime2Luv , name ="RGBPrime2Luv") 
addColorTransform( f=vigra.colors.transform_RGBPrime2RGB , name ="RGBPrime2RGB")
addColorTransform( f=vigra.colors.transform_RGBPrime2XYZ , name ="RGBPrime2XYZ")
addColorTransform( f=vigra.colors.transform_RGBPrime2YPrimeCbCr , name ="RGBPrime2YPrimeCbCr")
addColorTransform( f=vigra.colors.transform_RGBPrime2YPrimeIQ , name ="RGBPrime2YPrimeIQ")
addColorTransform( f=vigra.colors.transform_RGBPrime2YPrimePbPr , name ="RGBPrime2YPrimePbPr")
addColorTransform( f=vigra.colors.transform_RGBPrime2YPrimeUV , name ="RGBPrime2YPrimeUV")
addColorTransform( f=vigra.colors.transform_XYZ2Lab , name ="XYZ2Lab")
addColorTransform( f=vigra.colors.transform_XYZ2Luv , name ="YZ2Luv")
addColorTransform( f=vigra.colors.transform_XYZ2RGB , name ="XYZ2RGB")
addColorTransform( f=vigra.colors.transform_XYZ2RGBPrime , name ="XYZ2RGBPrime")
addColorTransform( f=vigra.colors.transform_YPrimeCbCr2RGBPrime , name ="YPrimeCbCr2RGBPrime")
addColorTransform( f=vigra.colors.transform_YPrimeIQ2RGBPrime , name ="YPrimeIQ2RGBPrime")
addColorTransform( f=vigra.colors.transform_YPrimePbPr2RGBPrime , name ="YPrimePbPr2RGBPrime")
addColorTransform( f=vigra.colors.transform_YPrimeUV2RGBPrime , name ="YPrimeUV2RGBPrime")
addColorTransform( f=vigra.colors.transform_sRGB2RGB , name ="sRGB2RGB")

 #######################################################
# 
#   FILTERS
#
########################################################

def _powerdGaussianSmoothing(img,sigma,power):
    img=img**power
    result = vigra.filters.gaussianSmoothing(img,sigma=sigma)
    result = result**(1.0/float(power))
    return result


node = vigraNode(
    nodeName="GaussianGradient",
    uiTemplate=[  ('sigma',  'spin', {'value': 1.00, 'step': 0.20, 'range': [0.10, None]}) ],
    f=vigra.filters.gaussianGradient,
    doChannelWise=True
)
fclib.registerNodeType(node,[('Image-Filters',)])


node = vigraNode(
    nodeName="GaussianGradientMagnitude",
    uiTemplate=[  ('sigma',  'spin', {'value': 1.00, 'step': 0.20, 'range': [0.10, None]}) ],
    f=vigra.filters.gaussianGradientMagnitude
)
fclib.registerNodeType(node,[('Image-Filters',)])
    

node = vigraNode(
    nodeName="GaussianSmoothing",
    uiTemplate=[  ('sigma',  'spin', {'value': 1.00, 'step': 0.10, 'range': [0.10, None]}) ],
    f=vigra.filters.gaussianSmoothing
)
fclib.registerNodeType(node,[('Image-Filters',)])

node = vigraNode(
    nodeName="PowerdGaussianSmoothing",
    uiTemplate=[  
        ('sigma',  'spin', {'value': 1.00, 'step': 0.10, 'range': [0.10, None]}) ,
        ('power',  'spin', {'value': 2.00, 'step': 1.00, 'range':  [0.001, None]})
    ],
    f=_powerdGaussianSmoothing
)
fclib.registerNodeType(node,[('Image-Filters',)])


node = vigraNode(
    nodeName="NonLinearDiffusion",
    uiTemplate=[ 
        ('edgeThreshold',  'spin', {'value': 0.25, 'step': 0.05, 'range': [0.01, None]}) ,
        ('scale',          'spin', {'value': 1.50, 'step': 0.25, 'range': [0.01, None]}) 
    ],
    f=vigra.filters.nonlinearDiffusion
)
fclib.registerNodeType(node,[('Image-Filters',)])

node = vigraNode(
    nodeName="RadialSymmetryTransform",
    uiTemplate=[  ('b',  'spin', {'value': 1.00, 'step': 0.10, 'range': [0.10, None]}) ],
    f=vigra.filters.radialSymmetryTransform2D,
    doChannelWise=True
)
fclib.registerNodeType(node,[('Image-Filters',)])

node = vigraNode(
    nodeName="RieszTransformOfLOG",
    uiTemplate=[  
        ('scale',  'spin', {'value': 1.00, 'step': 0.10, 'range': [0.10, None]}),
        ('xorder','intSpin', {'value': 1, 'min': 0, 'max': 1e9 }),
        ('yorder','intSpin', {'value': 1, 'min': 0, 'max': 1e9 })
    ],
    f=vigra.filters.rieszTransformOfLOG2D,
        doChannelWise=True
)
fclib.registerNodeType(node,[('Image-Filters',)])

node = vigraNode(
    nodeName="SimpleSharpening",
    uiTemplate=[  
        ('sharpeningFactor',  'spin', {'value': 1.00, 'step': 0.10, 'range': [0.10, None]})
    ],
    f=vigra.filters.simpleSharpening2D,
)
fclib.registerNodeType(node,[('Image-Filters',)])


node = vigraNode(
    nodeName="SymmetricGradient",
    uiTemplate=[  
        ('step_size','intSpin', {'value': 1, 'min': 1, 'max': 1e9 }),
    ],
    f=vigra.filters.symmetricGradient,
)
fclib.registerNodeType(node,[('Image-Filters',)])


#######################################################
# 
#   RECURSIVE - FILTERS
#
########################################################
node = vigraNode(
    nodeName="RecursiveFilter",
    uiTemplate=[  ('b',  'spin', {'value': 0.5, 'step': 0.10, 'range': [-0.999999999,0.999999999 ]}) ],
    f=vigra.filters.recursiveFilter2D,
)
fclib.registerNodeType(node,[('Image-Recursive-Filters',)])


node = vigraNode(
    nodeName="RecursiveGaussianSmoothing2D",
    uiTemplate=[  ('sigma',  'spin', {'value': 1.5, 'step': 0.10, 'range': [0.01,None]}) ],
    f=vigra.filters.recursiveGaussianSmoothing2D,
    doChannelWise=False
)
fclib.registerNodeType(node,[('Image-Recursive-Filters',)])


""" BUG IN WRAPPER!!! (OR DOC BUT IT IS WRAPPER)
node = vigraNode(
    nodeName="RecursiveGradient2D",
    uiTemplate=[  ('image',  'spin', {'value': 1.5, 'step': 0.10, 'range': [0.01,None]}) ],
    f=vigra.filters.recursiveGradient,
)
fclib.registerNodeType(node,[('Image-Recursive-Filters',)])
"""

node = vigraNode(
    nodeName="RecursiveLaplacian",
    uiTemplate=[  ('scale',  'spin', {'value': 1.5, 'step': 0.10, 'range': [0.01,None]}) ],
    f=vigra.filters.recursiveLaplacian2D,
)
fclib.registerNodeType(node,[('Image-Recursive-Filters',)])


node = vigraNode(
    nodeName="RecursiveSmooth",
    uiTemplate=[  ('scale',  'spin', {'value': 1.5, 'step': 0.10, 'range': [0.01,None]}) ],
    f=vigra.filters.recursiveSmooth2D,
)
fclib.registerNodeType(node,[('Image-Recursive-Filters',)])



#######################################################
# 
#   TENSOR
#
########################################################

def _tensorEigenvalues(tensor,sortEigenValues,eigenvalue=0):

    ew = vigra.filters.tensorEigenvalues(tensor)
    if  sortEigenValues :
        ew = np.sort(ew,axis=2)
    if eigenvalue<=1:
        return ew[:,:,eigenvalue]
    else :
        return  ew

node = vigraNode(
    nodeName="BoundaryTensor",
    uiTemplate=[('scale',          'spin', {'value': 1.50, 'step': 0.25, 'range': [0.01, None]})],
    f=vigra.filters.boundaryTensor2D
)
fclib.registerNodeType(node,[('Image-Tensors',)])


node = vigraNode(
    nodeName="StructureTensor",
    uiTemplate=[
        ('innerScale',          'spin', {'value': 1.50, 'step': 0.25, 'range': [0.01, None]}),
        ('outerScale',          'spin', {'value': 2.50, 'step': 0.25, 'range': [0.01, None]}),
    ],
    f=vigra.filters.structureTensor
)
fclib.registerNodeType(node,[('Image-Tensors',)])

node = vigraNode(
    nodeName="LaplacianOfGaussian",
    uiTemplate=[
        ('scale','spin', {'value': 1.50, 'step': 0.25, 'range': [0.01, None]})
    ],
    f=vigra.filters.laplacianOfGaussian
)
fclib.registerNodeType(node,[('Image-Tensors',)])


node = vigraNode(
    nodeName="TensorEigenvalues",
    uiTemplate=[
        ('sortEigenValues', 'check', {'value': True}),
        ('eigenvalue', 'combo', {'values': ['first', 'second','both'], 'index': 0})
    ],
    f=_tensorEigenvalues
)

fclib.registerNodeType(node,[('Image-Tensors',)])


class StructureTensorTrace(CtrlNode):
    """ structure tensor trace"""
    nodeName = "StructureTensorTrace"

    uiTemplate=[('innerscale', 'spin', {'value': 1.5, 'step': 0.1, 'range': [0.01, None]}),
                ('outerscale', 'spin', {'value': 2.5, 'step': 0.1, 'range': [0.01, None]})]
    def __init__(self, name):
        terminals = {
            'Image': dict(io='in'),     
            'dataOut': dict(io='out')  # to specify whether it is input or output
        }                              # other more advanced options are available
                                       # as well..
        CtrlNode.__init__(self, name, terminals=terminals)
    def process(self, Image, display=True):
        tensorValues = vigra.filters.structureTensor(image=Image,
                                                     innerScale = self.ctrls['innerscale'].value(),
                                                     outerScale = self.ctrls['outerscale'].value())
        out = np.zeros(tensorValues.shape[:2])
        stepSize = math.sqrt(2*tensorValues.shape[2] + 0.25) - 0.5
        assert stepSize == math.floor(stepSize)
        index = 0
        while (stepSize > 0):
            out += tensorValues[..., index]
            index += stepSize
            stepSize -= 1
        return {'dataOut': out}

        
fclib.registerNodeType(StructureTensorTrace ,[('Image-Tensors',)])



class TensorTrace(CtrlNode):
    """ calculate trace of tensor input """
    nodeName = "TensorTrace"

    def __init__(self, name):
        terminals = {
            'Tensor': dict(io='in'),
            'dataOut': dict(io='out')
        }

        CtrlNode.__init__(self, name, terminals=terminals)
    def process(self, Tensor, display=True):
        assert len(Tensor.shape) == 3
        out = np.zeros(Tensor.shape[:2])
        stepSize = math.sqrt(2*Tensor.shape[2] + 0.25) - 0.5
        assert stepSize == math.floor(stepSize)
        index = 0
        while (stepSize > 0):
            out += Tensor[..., index]
            index += stepSize
            stepSize -= 1
        return {'dataOut': out}


fclib.registerNodeType(TensorTrace, [('Image-Tensors',)])



#######################################################
# 
#   DISK-Filters
#
########################################################
node = vigraNode(
    nodeName="DiscClosing",
    uiTemplate=[
        ('radius','intSpin', {'value': 1, 'min': 1, 'max': 1e9 })
    ],
    f=vigra.filters.discClosing,
    dtypeIn=np.uint8
)
fclib.registerNodeType(node,[('Image-Disc-Filters',)])


node = vigraNode(
    nodeName="DiscDilation",
    uiTemplate=[
        ('radius','intSpin', {'value': 1, 'min': 1, 'max': 1e9 })
    ],
    f=vigra.filters.discDilation,
    dtypeIn=np.uint8
)
fclib.registerNodeType(node,[('Image-Disc-Filters',)])
    
node = vigraNode(
    nodeName="DiscErosion",
    uiTemplate=[
        ('radius','intSpin', {'value': 1, 'min': 1, 'max': 1e9 })
    ],
    f=vigra.filters.discErosion,
    dtypeIn=np.uint8
)
fclib.registerNodeType(node,[('Image-Disc-Filters',)])


node = vigraNode(
    nodeName="DiscMedian",
    uiTemplate=[
        ('radius','intSpin', {'value': 1, 'min': 1, 'max': 1e9 })
    ],
    f=vigra.filters.discMedian,
    dtypeIn=np.uint8
)
fclib.registerNodeType(node,[('Image-Disc-Filters',)])



node = vigraNode(
    nodeName="DiscOpening",
    uiTemplate=[
        ('radius','intSpin', {'value': 1, 'min': 1, 'max': 1e9 })
    ],
    f=vigra.filters.discOpening,
    dtypeIn=np.uint8
)
fclib.registerNodeType(node,[('Image-Disc-Filters',)])


node = vigraNode(
    nodeName="DiscRankOrderFilter",
    uiTemplate=[
        ('radius','intSpin', {'value': 1, 'min': 1, 'max': 1e9 }),
        ('rank','spin', {'value': 0.50, 'step': 0.1, 'range': [0.0, 1.0]})
    ],
    f=vigra.filters.discRankOrderFilter,
    dtypeIn=np.uint8
)
fclib.registerNodeType(node,[('Image-Disc-Filters',)])

"""
node = vigraNode(
    nodeName="HourGlassFilter",
    uiTemplate=[
        ('scale','spin', {'value': 1.50, 'step': 0.25, 'range': [0.01, None]})
        ,
        ('rho',  'spin', {'value': 1.50, 'step': 0.25, 'range': [0.01, None]})
    ],
    f=vigra.filters.hourGlassFilter2D
)
fclib.registerNodeType(node,[('Image-Tensors',)])
"""






##################################################
#
#   CHANNELS
#
###################################################
class ChannelStacker(Node):
    """ blend images (weighted), normalize, if neccessary """
    nodeName = "ChannelStacker"

    uiTemplate=[
    ]

    def __init__(self, name):
        """
        terminals = {
            'Data0': dict(io='in'),
            'Data1': dict(io='in'),
            'dataOut': dict(io='out'),
        }
        """

        terminals = OrderedDict()
        terminals['Input']=dict(io='in')
        terminals['dataOut']=dict(io='out')
        Node.__init__(self, name, terminals=terminals,allowAddInput=True)


    def process(self, *args,**kwargs):

        inputList = []
        inputShape = None
        # iterate over all terminals in ordere they where added
        for termName in self.terminals.keys():
            term = self.terminals[termName]
            if termName in self._inputs:
                print "kwargs[%s]"%termName, kwargs[termName]
                inputData  =  kwargs[termName]

                if inputData is not None:
                    data  = np.array(inputData,dtype=np.float32)
                    iShape = data.shape
                    reshapedInput = data.reshape([iShape[0],iShape[1],-1])
                    inputList.append(reshapedInput)
        
        if len(inputList)>0:
            stacked =  np.concatenate(inputList,axis=2)
            print "outShape",stacked.shape
        else:
            stacked = None
        return {'dataOut':stacked}

    def graphicsItem(self):
        if self._graphicsItem is None:
            self._graphicsItem = CustomNodeGraphicsItem(self,(150,200))
        return self._graphicsItem

fclib.registerNodeType(ChannelStacker, [('Image-Channels',)])

def _channelAcc(image,accumulation):

    if(accumulation==0):
        return np.sum(image,axis=2)
    elif(accumulation==1):
        return np.product(image,axis=2)
    elif(accumulation==2):
        return np.min(image,axis=2)
    elif(accumulation==3):
        return np.max(image,axis=2)
    elif(accumulation==4):
        return np.mean(image,axis=2)
    elif(accumulation==5):
        return np.median(image,axis=2)
    else:
        assert False

def _channelSelector(image,channel):
    return image[:,:,channel]


node = vigraNode(
    nodeName="ChannelAccumuation",
    uiTemplate=[
        ('accumulation', 'combo', {'values': ['sum', 'product','min','max','mean','median'], 'index': 0})
    ],
    f=_channelAcc
)

fclib.registerNodeType(node,[('Image-Channels',)])


node = vigraNode(
    nodeName="ChannelSelector",
    uiTemplate=[
       ('channel','intSpin', {'value': 0, 'min': 0, 'max': 1e9 })
    ],
    f=_channelSelector
)

fclib.registerNodeType(node,[('Image-Channels',)])

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


node = vigraNode(
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



node = vigraNode(
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



node = vigraNode(
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


node = vigraNode(
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


node = vigraNode(
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


node = vigraNode(
    nodeName="CornernessBeaudet",
    uiTemplate=[
       ('scale',     'spin',    {'value': 1.0, 'step': 0.5, 'range': [0.01, None]})
    ],
    f=vigra.analysis.cornernessBeaudet,
    doChannelWise=True
)
fclib.registerNodeType(node,[('Image-Analysis',)])



node = vigraNode(
    nodeName="CornernessBoundaryTensor",
    uiTemplate=[
       ('scale',     'spin',    {'value': 1.0, 'step': 0.5, 'range': [0.01, None]})
    ],
    f=vigra.analysis.cornernessBoundaryTensor,
    doChannelWise=True
)
fclib.registerNodeType(node,[('Image-Analysis',)])


node = vigraNode(
    nodeName="CornernessFoerstner",
    uiTemplate=[
       ('scale',     'spin',    {'value': 1.0, 'step': 0.5, 'range': [0.01, None]})
    ],
    f=vigra.analysis.cornernessFoerstner,
    doChannelWise=True
)
fclib.registerNodeType(node,[('Image-Analysis',)])


node = vigraNode(
    nodeName="CornernessHarris",
    uiTemplate=[
       ('scale',     'spin',    {'value': 1.0, 'step': 0.5, 'range': [0.01, None]})
    ],
    f=vigra.analysis.cornernessHarris,
    doChannelWise=True
)
fclib.registerNodeType(node,[('Image-Analysis',)])


node = vigraNode(
    nodeName="CornernessRohr",
    uiTemplate=[
       ('scale',     'spin',    {'value': 1.0, 'step': 0.5, 'range': [0.01, None]})
    ],
    f=vigra.analysis.cornernessRohr,
    doChannelWise=True
)
fclib.registerNodeType(node,[('Image-Analysis',)])
###################################################
#
#   Sampling
#
###################################################


def _resize(image,shapeX,shapeY,order,resizeType):
    if resizeType == 0 :
        return vigra.sampling.resize(image,shape=[shapeX,shapeY],order=order)
    elif resizeType == 1 :
       return vigra.sampling.resizeImageCatmullRomInterpolation(image,shape=[shapeX,shapeY])
    elif resizeType == 2 :
       return vigra.sampling.resizeImageCoscotInterpolation(image,shape=[shapeX,shapeY])
    elif resizeType == 3:
       return vigra.sampling.resizeImageLinearInterpolation(image,shape=[shapeX,shapeY])
    elif resizeType == 4 :
       return vigra.sampling.resizeImageNoInterpolation(image,shape=[shapeX,shapeY])
    elif resizeType == 5 :
       return vigra.sampling.resizeImageSplineInterpolation(image,shape=[shapeX,shapeY])


node = vigraNode(
    nodeName="ResampleImage",
    uiTemplate=[
        ('factor','spin', {'value': 2.0, 'step': 0.25, 'range': [0.0001, None]})
    ],
    f=vigra.sampling.resampleImage
)
fclib.registerNodeType(node,[('Image-Sampling',)])






node = vigraNode(
    nodeName="ResizeImage",
    uiTemplate=[
        ('shapeX','intSpin', {'value': 100, 'min': 1, 'max': 1e9 }),
        ('shapeY','intSpin', {'value': 100, 'min': 1, 'max': 1e9 }),
        ('order','intSpin', {'value': 3, 'min': 0, 'max': 3 }),
        ('resizeType', 'combo', {'values': ['Default','CatmullRom','CosCot','Linear','NoInterpolation','Spline'], 'index': 0})
    ],
    f=_resize
)
fclib.registerNodeType(node,[('Image-Sampling',)])








###################################################
#
#   NORMALIZE AND CLIP AND STUFF LIKE THAT
#
###################################################

def _normalize(data,newMin,newMax):
    d=data.copy()
    d-=data.min()
    d/=data.max()
    d*=(newMax-newMin)
    d+=newMin
    return d


def _normalizeQuantiles(data,newMin,newMax,quantileLow,quantileHigh):
    ql,qh = mquantiles(np.array(data), prob=[quantileLow,quantileHigh], alphap=0.4, betap=0.4, axis=None, limit=())

    d = np.clip(np.array(data),ql,qh)
    d-=data.min()
    d/=data.max()
    d*=(newMax-newMin)
    d+=newMin
    return d

node = vigraNode(
    nodeName="Normalize",
    uiTemplate=[
        ('newMin','spin', {'value': 0, 'step': 1, 'range': [None, None]}),
        ('newMax','spin', {'value': 255, 'step': 1, 'range': [None, None]})
    ],
    f=_normalize
)
fclib.registerNodeType(node,[('Image-Normalize',)])

node = vigraNode(
    nodeName="NormalizeQuantiles",
    uiTemplate=[
        ('newMin','spin', {'value': 0, 'step': 1, 'range': [None, None]}),
        ('newMax','spin', {'value': 255, 'step': 1, 'range': [None, None]}),
        ('quantileLow','spin', {'value':  0.1, 'step': 0.05, 'range': [0.0,1.0]} ),
        ('quantileHigh','spin', {'value': 0.9, 'step': 0.05, 'range': [0.0,1.0]} )
    ],
    f=_normalizeQuantiles
)
fclib.registerNodeType(node,[('Image-Normalize',)])








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
node = vigraNode(
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
#   NORMALIZE
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



node = vigraNode(
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

node = vigraNode(
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
#   Random Forest
#
###################################################


class RandomForest(CtrlNode):
    """ blend images (weighted), normalize, if neccessary """
    nodeName = "RandomForest"

    uiTemplate=[
        ('predictLabels', 'check',   {'value': True}),
        ('predictProbs', 'check',   {'value': True}),
        ('numTrees','intSpin', {'value': 10, 'min': 0, 'max': 1e9 })
    ]

    def __init__(self, name):
        terminals = {
            'Features': dict(io='in'),
            'Labels': dict(io='in'),
            'RF': dict(io='out'),
            'PredictedLabels': dict(io='out'),
            'PredictedProbs': dict(io='out'),
            'OOB-Error': dict(io='out'),
        }
        CtrlNode.__init__(self, name, terminals=terminals)
    def process(self, Features, Labels, display=True):
        numFeatures = Features.shape[2]

        shape = Labels.shape
        features = Features.astype(np.float32).reshape([-1,numFeatures])
        labels   = Labels.astype(np.uint32).reshape(-1)


        where1 = np.where(labels==1)
        where2 = np.where(labels==2)

        print where1
        print where2

        f1  = features[where1[0],:]
        f2  = features[where2[0],:]
        f= np.concatenate([f1,f2],axis=0).astype(np.float32)

        print "f1 shape",f1.shape
        print "f2 shape",f2.shape
        print "f shape",f.shape

        numLabeledPoints = f.shape[0]

        print "numLabeledPoints",numLabeledPoints

        l= np.zeros([f.shape[0],1],dtype=np.uint32)
        l[0:len(where1[0])]=0
        l[len(where1[0]):len(where1[0])+len(where2[0])]=1

        print "learn rf"
        print f.shape , l .shape 
        rf=vigra.learning.RandomForest(treeCount=self.ctrls['numTrees'].value())
        oob = rf.learnRF(f,l)
        print "learn rf done",oob

        print "predict"
        probs = rf.predictProbabilities(features)
        print "predict done"
        probsImg = np.array(probs)
        probsImg = probsImg.reshape(tuple(shape)+(2,))

        return {
            'RF': None,
            'PredictedLabels' : None,
            'PredictedProbs'  : probsImg,
            'OOB-Error'       : None
        }

    def graphicsItem(self):
        if self._graphicsItem is None:
            self._graphicsItem = CustomNodeGraphicsItem(self,(150,150))
        return self._graphicsItem


fclib.registerNodeType(RandomForest, [('Image-MachineLearning',)])



###################################################
#
#   numpy.where
#
###################################################
class NumpyWhere(CtrlNode):
    """ blend images (weighted), normalize, if neccessary """
    nodeName = "numpy.where"

    uiTemplate=[('ignore', 'spin', {'value' : 0, 'step' : 1, 'range': [0, None]})]

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
fclib.registerNodeType(NumpyWhere, [('Misc',)])



###################################################
#
#   FeatureStackToMatrix
#
###################################################
class FeatureStackToMatrix(CtrlNode):
    """ create feature matrix from labels and feature image """
    nodeName = "FeatureStackToMatrix"

    uiTemplate = [('no label', 'spin', {'value': 0, 'step' : 1, 'range' : [0, None]})]

    def __init__(self, name):
        terminals = {
            'LabelImage': dict(io='in'),
            'Features': dict(io='in'),
            'FeatureMatrix': dict(io='out'),
            'LabelVector': dict(io='out')
        }
        CtrlNode.__init__(self, name, terminals=terminals)
    def process(self, LabelImage, Features, display=True):
        if Features is None or \
           LabelImage.shape[0] == 0 or \
           np.all(LabelImage == 0):
            return {'FeatureMatrix': None, 'LabelVector': None}
        no_label_value = self.ctrls['no label'].value()

        # only for 2D -> workaraound for 3D?
        if len(Features.shape) == 2:
            Features = Features[..., np.newaxis]
        
        if LabelImage is not None:
            label_indices = np.where(LabelImage[...] != no_label_value)
            label_vector = LabelImage[label_indices]
            label_vector = label_vector[..., np.newaxis]
        else:
            label_indices = (slice(0, Features.shape[0]),
                             slice(0, Features.shape[1]))
            label_vector = None
        number_of_samples = label_indices[0].shape[0]
        number_of_features = Features.shape[2]
        feature_matrix = np.empty((number_of_samples, number_of_features))
        for index in xrange(Features.shape[-1]):
            feature = Features[..., index]
            feature_matrix[..., index] = feature[label_indices]
        return {'FeatureMatrix': feature_matrix, 'LabelVector': label_vector}
            
        
fclib.registerNodeType(FeatureStackToMatrix, [('Image-MachineLearning',)])

###################################################
#
#   numpy
#
###################################################
class NumpyRequire(CtrlNode):
    """ blend images (weighted), normalize, if neccessary """
    nodeName = "NumpyRequire"

    dtypes= [
        np.bool,
        np.uint8,
        np.uint16,
        np.uint32,
        np.uint64,
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.float32,
        np.float64
    ]

    dtypeStrs = [ str(d) for d in dtypes]
    uiTemplate = [
        ('dtype', 'combo', {'values': dtypeStrs})
    ]
    def __init__(self, name):
        terminals = {
            'dataIn': dict(io='in'),
            'dataOut': dict(io='out')
        }
        CtrlNode.__init__(self, name, terminals=terminals)
    def process(self, dataIn, display=True):

        return { 'dataOut': 
            np.require(dataIn,dtype=NumpyRequire.dtypes[  self.ctrls['dtype'].currentIndex()   ])
        }
fclib.registerNodeType(NumpyRequire, [('Numpy',)])
