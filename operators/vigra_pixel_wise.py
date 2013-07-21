import numpy as np
import vigra
import math
from node_base import numpyInNumpyOutNode
import pyqtgraph.flowchart.library as fclib
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

node = numpyInNumpyOutNode(
    nodeName="Brightness",
    uiTemplate=[  ('factor',  'spin', {'value': 1.00, 'step': 0.20, 'range': [0.10, None]}) ],
    f=_brightness,
)
fclib.registerNodeType(node,[('Image-Color/Intensity',)])

node = numpyInNumpyOutNode(
    nodeName="Contrast",
    uiTemplate=[  ('factor',  'spin', {'value': 1.00, 'step': 0.20, 'range': [0.10, None]}) ],
    f=_contrast,
)
fclib.registerNodeType(node,[('Image-Color/Intensity',)])

node = numpyInNumpyOutNode(
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
    node = numpyInNumpyOutNode(
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