import numpy as np
import vigra
import math
from collections import OrderedDict

from node_base import MyCtrlNode,numpyInNumpyOutNode
import pyqtgraph.flowchart.library as fclib





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
