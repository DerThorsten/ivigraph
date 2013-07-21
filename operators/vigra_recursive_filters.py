import numpy as np
import vigra
import math
from node_base import numpyInNumpyOutNode
import pyqtgraph.flowchart.library as fclib





#######################################################
# 
#   RECURSIVE - FILTERS
#
########################################################
node = numpyInNumpyOutNode(
    nodeName="RecursiveFilter",
    uiTemplate=[  ('b',  'spin', {'value': 0.5, 'step': 0.10, 'range': [-0.999999999,0.999999999 ]}) ],
    f=vigra.filters.recursiveFilter2D,
)
fclib.registerNodeType(node,[('Image-Recursive-Filters',)])


node = numpyInNumpyOutNode(
    nodeName="RecursiveGaussianSmoothing2D",
    uiTemplate=[  ('sigma',  'spin', {'value': 1.5, 'step': 0.10, 'range': [0.01,None]}) ],
    f=vigra.filters.recursiveGaussianSmoothing2D,
    doChannelWise=False
)
fclib.registerNodeType(node,[('Image-Recursive-Filters',)])


""" BUG IN WRAPPER!!! (OR DOC BUT IT IS WRAPPER)
node = numpyInNumpyOutNode(
    nodeName="RecursiveGradient2D",
    uiTemplate=[  ('image',  'spin', {'value': 1.5, 'step': 0.10, 'range': [0.01,None]}) ],
    f=vigra.filters.recursiveGradient,
)
fclib.registerNodeType(node,[('Image-Recursive-Filters',)])
"""

node = numpyInNumpyOutNode(
    nodeName="RecursiveLaplacian",
    uiTemplate=[  ('scale',  'spin', {'value': 1.5, 'step': 0.10, 'range': [0.01,None]}) ],
    f=vigra.filters.recursiveLaplacian2D,
)
fclib.registerNodeType(node,[('Image-Recursive-Filters',)])


node = numpyInNumpyOutNode(
    nodeName="RecursiveSmooth",
    uiTemplate=[  ('scale',  'spin', {'value': 1.5, 'step': 0.10, 'range': [0.01,None]}) ],
    f=vigra.filters.recursiveSmooth2D,
)
fclib.registerNodeType(node,[('Image-Recursive-Filters',)])
