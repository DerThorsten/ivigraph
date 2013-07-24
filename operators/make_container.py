import numpy as np
import vigra
import math
from collections import OrderedDict

from node_base import MyNode,MyCtrlNode,numpyInNumpyOutNode
import pyqtgraph.flowchart.library as fclib



##################################################
#
#   MakeContainer
#
###################################################
class MakeList(MyNode):
    """ make a list from input(s) """
    nodeName = "MakeList"

    uiTemplate=[
    ]

    def __init__(self, name):

        terminals = OrderedDict()
        terminals['Input']=dict(io='in')
        terminals['dataOut']=dict(io='out')
        MyNode.__init__(self, name, terminals=terminals,nodeSize=(100,150),allowAddInput=True)


    def execute(self, *args,**kwargs):
        inputList = []
        # iterate over all terminals in ordere they where added
        for termName in self.terminals.keys():
            term = self.terminals[termName]
            if termName in self._inputs:
                inputData  =  kwargs[termName]
                if inputData is not None:
                    inputList.append(inputData)
        
        return {'dataOut':inputList}


fclib.registerNodeType(MakeList, [('Container',)])


class MakeTuple(MyNode):
    """ make a tuple from input(s) """
    nodeName = "MakeTuple"

    uiTemplate=[
    ]

    def __init__(self, name):

        terminals = OrderedDict()
        terminals['Input']=dict(io='in')
        terminals['dataOut']=dict(io='out')
        MyNode.__init__(self, name, terminals=terminals,nodeSize=(100,150),allowAddInput=True)


    def execute(self, *args,**kwargs):
        inputList = []
        # iterate over all terminals in ordere they where added
        for termName in self.terminals.keys():
            term = self.terminals[termName]
            if termName in self._inputs:
                inputData  =  kwargs[termName]
                if inputData is not None:
                    inputList.append(inputData)
        
        return {'dataOut':tuple(inputList)}


fclib.registerNodeType(MakeTuple, [('Container',)])



