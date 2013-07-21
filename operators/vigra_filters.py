import numpy as np
import vigra
import math
from node_base import numpyInNumpyOutNode
import pyqtgraph.flowchart.library as fclib




#######################################################
# 
#   FILTERS
#
########################################################

# ERROR!!!
"""
node = numpyInNumpyOutNode(
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

def _powerdGaussianSmoothing(img,sigma,power):
    img=img**power
    result = vigra.filters.gaussianSmoothing(img,sigma=sigma)
    result = result**(1.0/float(power))
    return result


node = numpyInNumpyOutNode(
    nodeName="GaussianGradient",
    uiTemplate=[  ('sigma',  'spin', {'value': 1.00, 'step': 0.20, 'range': [0.10, None]}) ],
    f=vigra.filters.gaussianGradient,
    doChannelWise=True
)
fclib.registerNodeType(node,[('Image-Filters',)])


node = numpyInNumpyOutNode(
    nodeName="GaussianGradientMagnitude",
    uiTemplate=[  ('sigma',  'spin', {'value': 1.00, 'step': 0.20, 'range': [0.10, None]}) ],
    f=vigra.filters.gaussianGradientMagnitude
)
fclib.registerNodeType(node,[('Image-Filters',)])
    

node = numpyInNumpyOutNode(
    nodeName="GaussianSmoothing",
    uiTemplate=[  ('sigma',  'spin', {'value': 1.00, 'step': 0.10, 'range': [0.10, None]}) ],
    f=vigra.filters.gaussianSmoothing
)
fclib.registerNodeType(node,[('Image-Filters',)])

node = numpyInNumpyOutNode(
    nodeName="PowerdGaussianSmoothing",
    uiTemplate=[  
        ('sigma',  'spin', {'value': 1.00, 'step': 0.10, 'range': [0.10, None]}) ,
        ('power',  'spin', {'value': 2.00, 'step': 1.00, 'range':  [0.001, None]})
    ],
    f=_powerdGaussianSmoothing
)
fclib.registerNodeType(node,[('Image-Filters',)])


node = numpyInNumpyOutNode(
    nodeName="NonLinearDiffusion",
    uiTemplate=[ 
        ('edgeThreshold',  'spin', {'value': 0.25, 'step': 0.05, 'range': [0.01, None]}) ,
        ('scale',          'spin', {'value': 1.50, 'step': 0.25, 'range': [0.01, None]}) 
    ],
    f=vigra.filters.nonlinearDiffusion
)
fclib.registerNodeType(node,[('Image-Filters',)])

node = numpyInNumpyOutNode(
    nodeName="RadialSymmetryTransform",
    uiTemplate=[  ('b',  'spin', {'value': 1.00, 'step': 0.10, 'range': [0.10, None]}) ],
    f=vigra.filters.radialSymmetryTransform2D,
    doChannelWise=True
)
fclib.registerNodeType(node,[('Image-Filters',)])

node = numpyInNumpyOutNode(
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

node = numpyInNumpyOutNode(
    nodeName="SimpleSharpening",
    uiTemplate=[  
        ('sharpeningFactor',  'spin', {'value': 1.00, 'step': 0.10, 'range': [0.10, None]})
    ],
    f=vigra.filters.simpleSharpening2D,
)
fclib.registerNodeType(node,[('Image-Filters',)])


node = numpyInNumpyOutNode(
    nodeName="SymmetricGradient",
    uiTemplate=[  
        ('step_size','intSpin', {'value': 1, 'min': 1, 'max': 1e9 }),
    ],
    f=vigra.filters.symmetricGradient,
)
fclib.registerNodeType(node,[('Image-Filters',)])