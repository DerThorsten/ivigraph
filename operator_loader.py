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

from collections import OrderedDict
from nodegraphics import CustomNodeGraphicsItem

from termcolor import colored
import time


# load disk filters

from operators.node_base import convertNh, MyCtrlNode,numpyInNumpyOutNode

import operators.vigra_disk_filters
import operators.vigra_pixel_wise
import operators.vigra_filters
import operators.vigra_recursive_filters
import operators.vigra_tensors
import operators.vigra_analysis
import operators.vigra_sampling
import operators.vigra_machine_learning
from operators.normalize import _normalize
from operators.node_base import MyCtrlNode

import operators.vigra_segmentation 


import operators.numpy_op           #TODO
import operators.channels
import operators.normalize











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
















