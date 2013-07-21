import numpy as np
import vigra
import math
from collections import OrderedDict

from node_base import convertNh, MyNode,MyCtrlNode,numpyInNumpyOutNode
import pyqtgraph.flowchart.library as fclib






###################################################
#
#   numpy
#
###################################################

_numpyDtypes= [
    np.float32,
    np.float64,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.bool
]

_numpyDtypeStrs = [ str(d) for d in _numpyDtypes]


_stringedBooleanOperators = {
    '=='    : '__eq__' , 
    '!='    : '__ne__' , 
    '<'     : '__lt__' , 
    '>'     : '__gt__' , 
    '<='    : '__le__' , 
    '>='    : '__ge__' 
}

class NumpyRequire(MyCtrlNode):
    """ blend images (weighted), normalize, if neccessary """
    nodeName = "numpy.require"
    uiTemplate = [
        ('dtype', 'combo', {'values': _numpyDtypeStrs})
    ]
    def __init__(self, name):
        terminals = {
            'dataIn': dict(io='in'),
            'dataOut': dict(io='out')
        }
        MyCtrlNode.__init__(self, name, terminals=terminals)
    def execute(self, dataIn, display=True):

        return { 'dataOut': 
            np.require(dataIn,dtype=_numpyDtypes[  self.ctrls['dtype'].currentIndex()   ])
        }

fclib.registerNodeType(NumpyRequire, [('Numpy',)])





class NumpyWhereNot(MyCtrlNode):
    """ blend images (weighted), normalize, if neccessary """
    nodeName = "numpy.whereNot"

    uiTemplate=[('isNotValue', 'spin', {'value' : 0, 'step' : 1, 'range': [0, None]})]

    def __init__(self, name):
        terminals = {
            'Image': dict(io='in'),
            'Indices': dict(io='out')
        }
        MyCtrlNode.__init__(self, name, terminals=terminals)
    def execute(self, Image, display=True):
        ignore_value = self.ctrls['ignore'].value()
        if ignore_value is not None:
            return {'Indices': np.where(Image != ignore_value)}
        else:
            return {'Indices': None}
fclib.registerNodeType(NumpyWhereNot, [('Numpy',)])
