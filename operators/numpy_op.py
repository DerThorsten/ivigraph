import numpy as np
import vigra
import math
from collections import OrderedDict

from node_base import AdvCtrlNode,convertNh, MyNode,MyCtrlNode,numpyInNumpyOutNode
import pyqtgraph.flowchart.library as fclib


import pyqtgraph.parametertree.parameterTypes as pTypes
from pyqtgraph.parametertree import Parameter, ParameterTree, ParameterItem, registerParameterType







class TestNode(AdvCtrlNode):
    """ blend images (weighted), normalize, if neccessary """
    nodeName = "TestNode"

    def __init__(self, name):
        terminals = {
            'dataIn': dict(io='in'),
            'dataOut': dict(io='out')
        }
        params = [
            {'name': 'Group1', 'type': 'group', 'children': [
                {'name': 'NamedList_', 'type': 'list', 'values': {"one": 1, "two": 2, "three": 3}, 'value': 2},
                {'name': 'NamedList', 'type': 'list', 'values': {"one": 1, "two": 2, "three": 3}, 'value': 2},
                {'name': 'Subgroup', 'type': 'group', 'children': [
                    {'name': 'Sub-param1', 'type': 'int', 'value': 10},
                    {'name': 'Sub-param2', 'type': 'float', 'value': 1.2e6},
                    {'name': 'Sub-param3', 'type': 'float', 'value': 1.2e6},
                ]},
                {'name': 'Action Parameter', 'type': 'action'},
            ]},
            {'name': 'P2', 'type': 'int', 'value': 10},
        ]


        
        AdvCtrlNode.__init__(self, name,ui=params, terminals=terminals)
    def execute(self, dataIn, display=True):
        print "\n\n\n-------1------\n\n\n",self.getParamValue(['P2'])
        print "\n\n\n-------2------\n\n\n",self.getParamValue(['Group1','NamedList_'])
        print "\n\n\n-------3------\n\n\n",self.getParamValue(['Group1','Subgroup','Sub-param2'])
        return {'dataOut':None}


fclib.registerNodeType(TestNode, [('Tests',)])


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
