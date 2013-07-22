import numpy as np
import vigra
import math
from collections import OrderedDict

from node_base import MyCtrlNode,numpyInNumpyOutNode
import pyqtgraph.flowchart.library as fclib






##################################################
#
#   SELECTOR
#
###################################################

class Selector(MyCtrlNode):
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
        MyCtrlNode.__init__(self, name, terminals=terminals)
    def execute(self, A,B, display=True):
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
