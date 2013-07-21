import numpy as np
import vigra
import math
from collections import OrderedDict

from node_base import convertNh, MyNode,MyCtrlNode,numpyInNumpyOutNode
import pyqtgraph.flowchart.library as fclib








        
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


node = numpyInNumpyOutNode(
    nodeName="ResampleImage",
    uiTemplate=[
        ('factor','spin', {'value': 2.0, 'step': 0.25, 'range': [0.0001, None]})
    ],
    f=vigra.sampling.resampleImage
)
fclib.registerNodeType(node,[('Image-Sampling',)])


node = numpyInNumpyOutNode(
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


