import numpy as np
import vigra
import math
from collections import OrderedDict
from scipy.stats.mstats import mquantiles

from node_base import convertNh, MyNode,MyCtrlNode,numpyInNumpyOutNode
import pyqtgraph.flowchart.library as fclib




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

node = numpyInNumpyOutNode(
    nodeName="Normalize",
    uiTemplate=[
        ('newMin','spin', {'value': 0, 'step': 1, 'range': [None, None]}),
        ('newMax','spin', {'value': 255, 'step': 1, 'range': [None, None]})
    ],
    f=_normalize
)
fclib.registerNodeType(node,[('Image-Normalize',)])

node = numpyInNumpyOutNode(
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