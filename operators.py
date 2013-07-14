from pyqtgraph.flowchart import Flowchart, Node
import pyqtgraph.flowchart.library as fclib
from pyqtgraph.flowchart.library.common import CtrlNode
from pyqtgraph.flowchart import Node
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg
import numpy
import scipy.ndimage
import vigra
import functools
from scipy.stats.mstats import mquantiles



def convertNh(nh):
    if nh == 0 : neighborhood = 4
    else: neighborhood =  8
    return neighborhood


def vigraNode(nodeName,uiTemplate,f,dtypeIn=numpy.float32,dtypeOut=numpy.float32,doChannelWise=False):
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
            dataInVigra = numpy.require(dataIn,dtype=dtypeIn)
            dataInVigra = numpy.squeeze(dataInVigra)
            

            if doChannelWise == False or dataInVigra.ndim==2 or dataInVigra.shape[2]==1:
                print "Single Input  ",dataInVigra.shape,dataInVigra.dtype
                vigraResult = f(dataInVigra,**kwargs)
            else:

                numChannels = dataInVigra.shape[2]
                vigraResult  = numpy.ones( dataInVigra.shape,dtype=dtypeIn)
                for c in range(numChannels):
                    print "channel wise input :",dataInVigra[:,:,c].shape,dataInVigra[:,:,c].dtype
                    vigraResult[:,:,c]=f(dataInVigra[:,:,c],**kwargs)
            vigraResult = numpy.squeeze(vigraResult)
            vigraResult = numpy.require(vigraResult,dtype=dtypeOut)


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
        ew = numpy.sort(ew,axis=2)
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
    dtypeIn=numpy.uint8
)
fclib.registerNodeType(node,[('Image-Disc-Filters',)])


node = vigraNode(
    nodeName="DiscDilation",
    uiTemplate=[
        ('radius','intSpin', {'value': 1, 'min': 1, 'max': 1e9 })
    ],
    f=vigra.filters.discDilation,
    dtypeIn=numpy.uint8
)
fclib.registerNodeType(node,[('Image-Disc-Filters',)])
    
node = vigraNode(
    nodeName="DiscErosion",
    uiTemplate=[
        ('radius','intSpin', {'value': 1, 'min': 1, 'max': 1e9 })
    ],
    f=vigra.filters.discErosion,
    dtypeIn=numpy.uint8
)
fclib.registerNodeType(node,[('Image-Disc-Filters',)])


node = vigraNode(
    nodeName="DiscMedian",
    uiTemplate=[
        ('radius','intSpin', {'value': 1, 'min': 1, 'max': 1e9 })
    ],
    f=vigra.filters.discMedian,
    dtypeIn=numpy.uint8
)
fclib.registerNodeType(node,[('Image-Disc-Filters',)])



node = vigraNode(
    nodeName="DiscOpening",
    uiTemplate=[
        ('radius','intSpin', {'value': 1, 'min': 1, 'max': 1e9 })
    ],
    f=vigra.filters.discOpening,
    dtypeIn=numpy.uint8
)
fclib.registerNodeType(node,[('Image-Disc-Filters',)])


node = vigraNode(
    nodeName="DiscRankOrderFilter",
    uiTemplate=[
        ('radius','intSpin', {'value': 1, 'min': 1, 'max': 1e9 }),
        ('rank','spin', {'value': 0.50, 'step': 0.1, 'range': [0.0, 1.0]})
    ],
    f=vigra.filters.discRankOrderFilter,
    dtypeIn=numpy.uint8
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

def _channelAcc(image,accumulation):

    if(accumulation==0):
        return numpy.sum(image,axis=2)
    elif(accumulation==1):
        return numpy.product(image,axis=2)
    elif(accumulation==2):
        return numpy.min(image,axis=2)
    elif(accumulation==3):
        return numpy.max(image,axis=2)
    elif(accumulation==4):
        return numpy.mean(image,axis=2)
    elif(accumulation==5):
        return numpy.median(image,axis=2)
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

    resX=numpy.zeros(img.shape)
    resY=numpy.zeros(img.shape)
    if(extrema==0):
        for x in range(img.shape[0]):
            if(x>0 and x<img.shape[0]-1):

                A = img[x-1,:]
                B = img[x  ,:]
                C = img[x+1,:]
                R1 = numpy.array(B < A).astype(numpy.float32)
                R2 = numpy.array(B < C).astype(numpy.float32)
                resX[x,:]=R1*R2
        for y in range(img.shape[1]):
            if(y>0 and y<img.shape[1]-1):
                A = img[:,y-1]
                B = img[:,y  ]
                C = img[:,y+1]
                R1 = numpy.array(B < A).astype(numpy.float32)
                R2 = numpy.array(B < C).astype(numpy.float32)
                resY[:,y]=R1*R2

    else :
        for x in range(img.shape[0]):
            if(x>0 and x<img.shape[0]-1):

                A = img[x-1,:]
                B = img[x  ,:]
                C = img[x+1,:]
                R1 = numpy.array(B > A).astype(numpy.float32)
                R2 = numpy.array(B > C).astype(numpy.float32)
                resX[x,:]=R1*R2
        for y in range(img.shape[1]):
            if(y>0 and y<img.shape[1]-1):
                A = img[:,y-1]
                B = img[:,y  ]
                C = img[:,y+1]
                R1 = numpy.array(B > A).astype(numpy.float32)
                R2 = numpy.array(B > C).astype(numpy.float32)
                resY[:,y]=R1*R2
    res = resX+resY 
    res[numpy.where(res>1)]=1.0
    return res*2.0


node = vigraNode(
    nodeName="ExtremaLines",
    uiTemplate=[
        ('extrema', 'combo', {'values': ['min', 'max'], 'index': 0})
    ],
    f=_kammlinie,
    doChannelWise=True,
    dtypeOut=numpy.uint32,
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
    dtypeOut=numpy.uint32,
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
    dtypeIn=numpy.float32,
    dtypeOut=numpy.uint32
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
    ql,qh = mquantiles(numpy.array(data), prob=[quantileLow,quantileHigh], alphap=0.4, betap=0.4, axis=None, limit=())

    d = numpy.clip(numpy.array(data),ql,qh)
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
    m_sepia = numpy.asarray([[0.393, 0.769, 0.189],
                             [0.349, 0.686, 0.168],
                             [0.272, 0.534, 0.131]])






    for y in range(data.shape[1]):
        for x in range(data.shape[0]):
            d[x,y,:]=numpy.dot(d[x,y,:],m_sepia.T)
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
    #superimg.visualize(imgRGBBig,cgp,numpy.ones(cgp.numCells(1),dtype=numpy.float32), cmap='jet',title='mixed')
   
    assert labels.shape[2] == 1
    labels = labels.squeeze()
    assert labels.ndim == 2, "labels has shape %r" % (labels.shape,)

    return labels,numseg


def _permuteLabels(data):

    flat = numpy.array(data).reshape([-1])
    unique , relabeling = numpy.unique(flat,return_inverse=True)
    permUnique = numpy.random.permutation(unique)
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
#   USER NODE
#
###################################################
class Paint(CtrlNode):
    """ (seeded) watershed"""
    nodeName = "Paint"

    uiTemplate=[
        ('eventMode', 'combo', {'values': ['recordClicks', 'clearAndIgnore'], 'index': 0}),
        ('FlipForUpdate', 'check',   {'value': False})

    ]
    def __init__(self, name):
        terminals = {
            'dataIn': dict(io='in'),
            'view': dict(io='in'),     
            'dataOut': dict(io='out')  # to specify whether it is input or output
        }                              # other more advanced options are available
                                       # as well..
        CtrlNode.__init__(self, name, terminals=terminals)
    def process(self, dataIn,view, display=True):
        dataOut=dataIn.copy()
        recordClicks = self.ctrls['eventMode'].currentIndex()    

        if (recordClicks==0):
            print "clicks in node",len(view.clicks)

            for x,y in view.clicks:
                dataOut[x,y,:]=0.0
        else:
            print "reset node"
            view.clicks=[]
        return {'dataOut':dataOut}
        

        
fclib.registerNodeType(Paint ,[('Image-Paint',)])