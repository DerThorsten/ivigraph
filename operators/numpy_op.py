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


        clipOpts = {'name': 'Clipping', 'type': 'group', 'children': [   

            {'name': 'clipQuantiles',       'type': 'bool',     'value':  False},
            {'name': 'clipChannelWise',     'type': 'bool',     'value':  True},
            {'name': 'QuantileLow',         'type': 'float',    'value':  0.001},
            {'name': 'QuantileHigh',        'type': 'float',    'value':  0.999},
                    #{'name': 'Action Parameter', 'type': 'action'},
        ]}

        normalizationOpts = {'name': 'Normalization', 'type': 'group', 'children': [   
            {'name': 'normalize',      'type': 'bool',     'value':  False},
            {'name': 'normalizeChannelWise','type': 'bool',     'value':  True},
            {'name': 'minimum',             'type': 'float',    'value':  0.0},
            {'name': 'maximum',             'type': 'float',    'value':  1.0},
                    #{'name': 'Action Parameter', 'type': 'action'},
        ]}


        unNormalizationOpts = {'name': 'Normalization', 'type': 'group', 'children': [   
            {'name': 'normalize',      'type': 'bool',     'value':  False},
            {'name': 'normalizeChannelWise','type': 'bool',     'value':  True},
            {'name': 'fromInputRange', 'type': 'list', 'values': {"disabled":0,"InputMinMax" : 1, "InputMean" : 2}, 'value': 0},
            {'name': 'minimum',             'type': 'float',    'value':  0.0},
            {'name': 'maximum',             'type': 'float',    'value':  1.0},
                    #{'name': 'Action Parameter', 'type': 'action'},
        ]}


        dtypeOpt = {'name': 'explicit dtype', 'type': 'list', 'values': 
            {
                    "float32" : np.float32,
                    "float64" : np.float64,
                    "uint8" : np.uint8,
                    "uint16" : np.uint16,
                    "uint32" : np.uint32,
                    "uint64" : np.uint64,
                    "int8" : np.int8,
                    "int16" : np.int16,
                    "int32" : np.int32,
                    "int64" : np.int64,
                    "bool" : np.bool
            }, 'value': np.float32
        }



        dtypeOpts = {'name': 'DtypeOpts', 'type': 'group', 'children': [
            {'name': 'dtype', 'type': 'list', 'values': {"asInput":0,"Explicit" : 1}, 'value': 0},
            dtypeOpt
        ]}




        params = [
            {'name': 'InputOptions', 'type': 'group', 'children':
                [
                    clipOpts,
                    normalizationOpts,
                    dtypeOpts,
                ]
            },

            {'name': 'NodeOptions', 'type': 'group', 'children':
                [
                    {'name': 'scale',           'type': 'float',    'value':  1.0},
                    {'name': 'edgeThreshold',   'type': 'float',    'value':  0.25},
                ]
            },

            {'name': 'OutputOptions', 'type': 'group', 'children':
                [
                    clipOpts,
                    unNormalizationOpts,
                    dtypeOpts,
                ]
            },
        ]



        
        AdvCtrlNode.__init__(self, name,ui=params, terminals=terminals)

        param = self.param
        print param.names
        for c in param.children():
            print c.name()
        print param.param('NodeOptions').__dict__
        param.param('NodeOptions').opts['expanded']=False
        param.param('NodeOptions').emitTreeChanges()
        #param.param('NodeOptions').hide()
        #self.uiTree.collapseTree(param.param('NodeOptions').makeTreeItem(1))
        #param.param('NodeOptions').hide()
        self.uiTree.update()
    def execute(self, dataIn, display=True):
        #print "1",self.getParamValue(['P2'])
        #print "2",self.getParamValue(['Group1','NamedList_'])
        #print "3",self.getParamValue(['Group1','Subgroup','Sub-param2'])
        #print "4",self.getParamValue(['Group1','Subgroup','Action Parameter'])
        return {'dataOut':None}


    def onActionButtonPress(self,actionPath):
        print "onActionButtonPress",actionPath


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
