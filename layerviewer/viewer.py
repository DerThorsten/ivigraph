import numpy as np
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg
from collections import OrderedDict

import pyqtgraph.parametertree.parameterTypes as pTypes
from pyqtgraph.parametertree import Parameter, ParameterTree, ParameterItem, registerParameterType

from helpers import setPolicy

from abc import ABCMeta

import layers 

def reusedSpaceCopy(old,new):
    if old is None:
        return new.copy()
    if tuple(old.shape)==tuple(new.shape):
        old[:]=new[:]
        return old
    else:
        return new.copy()



def setupLayerParam(name,paramList):
    basicParam = { 'name': name,'expanded':False, 'type': 'group',
                        'autoIncrementName':True, 'children':paramList }
    return basicParam




class LayerControlTree(QtGui.QWidget):
    def __init__(self, parent=None):
        QtGui.QWidget.__init__(self, parent)


        self.basicParam = {'name': 'Layer','expanded':False, 'type': 'group','autoIncrementName':True, 'children': [
            {'name': 'LayerType', 'type': 'list', 'values': {"RGB": 0, "LAB": 1, "ChannelWise": 2}, 'value': 0},
            {'name': 'Channel', 'type': 'int', 'value': 0,'limits':[0,2]},
            {'name': 'Opacity', 'type': 'float', 'value': 0.0, 'step': 0.1,'limits':[0,1]},
            {'name': 'Show', 'type': 'bool', 'value': True, 'tip': "Show / Hide this layer"},
            {'name': 'HideOthers', 'type': 'action','tip':"Hide all other layers"},
            {'name': 'Gradient', 'type': 'colormap'},
            #{'name': 'Subgroup', 'type': 'group', 'children': [
            #    {'name': 'Sub-param 1', 'type': 'int', 'value': 10},
            #    {'name': 'Sub-param 2', 'type': 'float', 'value': 1.2e6},
            #]},
            #{'name': 'Text Parameter', 'type': 'text', 'value': 'Some text...'},
            #{'name': 'Action Parameter', 'type': 'action'},
        ]}



        params = []
            

        self.paramGroup = Parameter.create(name='params', type='group', children=params)
        self.paramGroup.sigTreeStateChanged.connect(self.change)

        self.parameterTree = ParameterTree()
        self.parameterTree.setParameters(self.paramGroup, showTop=False)

        # add view box to graph view layout
        self.hbox = QtGui.QHBoxLayout()
        self.setLayout(self.hbox)
        self.hbox.addWidget(self.parameterTree)

    def addLayerControl(self,layer):
        layerParam = setupLayerParam(layer.layerName(),layer.controlTemplate())
        layerParam = self.paramGroup.addChild(layerParam)
        # pass parameter to layer itself
        layer.setLayerParameter(layerParam)

    def removeLayerControl(self,layer):
        paramToRemove = layer.layerParameter
        self.paramGroup.removeChild(paramToRemove)



    ## If anything changes in the tree, print a message
    def change(self,param, changes):
        #print("tree changes:")
        for param, change, data in changes:
            path = self.paramGroup.childPath(param)
            if path is not None:
                childName = '.'.join(path)
            else:
                childName = param.name()
            #print('  parameter: %s'% childName)
            #print('  change:    %s'% change)
            #print('  data:      %s'% str(data))
            #print('  ----------')

class LayerView(QtGui.QWidget):
    def __init__(self, parent=None):
        QtGui.QWidget.__init__(self, parent)


        self.graphView       = pg.GraphicsView()
        self.graphViewLayout = QtGui.QGraphicsGridLayout()
        self.graphView.centralWidget.setLayout(self.graphViewLayout)
        setPolicy(self.graphView,QtGui.QSizePolicy.Expanding)


        # view box
        self.viewBox = pg.ViewBox()
        self.viewBox.setAspectLocked(True)
        # add view box to graph view layout
        self.graphViewLayout.addItem(self.viewBox,0,0)
        self.hbox = QtGui.QHBoxLayout()
        self.setLayout(self.hbox)
        self.hbox.addWidget(self.graphView)

        # flip the view box
        self.viewBox.invertY(True)



class LayerViewer(QtGui.QWidget):



    def __init__(self,parent=None):


        QtGui.QWidget.__init__(self, parent)


        # set up window
        self.win = QtGui.QMainWindow()
        self.win.setWindowTitle("LayerViewerTitle")
        self.win.setCentralWidget(self)

        # set up main layout
        self.mainLayout = QtGui.QVBoxLayout()
        self.setLayout(self.mainLayout)
        
        # main splitter ( Vieer | Controll)
        self.viewControllSplitter = QtGui.QSplitter(QtCore.Qt.Horizontal)
        self.mainLayout.addWidget(self.viewControllSplitter)



        self.layerView    = LayerView(parent=self)
        self.layerControl = LayerControlTree(parent=self)

        # add layerView and layerConstrol to viewControllSplitter
        self.viewControllSplitter.addWidget(self.layerView)
        self.viewControllSplitter.addWidget(self.layerControl)

        # layers 
        self.layers = OrderedDict()

    def addLayer(self,name,layerType):
        # get the class of the layer
        layerClass  = layers.layerTypes[layerType]
        # construct the layer
        layer = layerClass(name=name,viewBox=self.layerView.viewBox)

        # add layer controll (this will setup the member/attribute
        # layer.layerParameter)
        self.layerControl.addLayerControl(layer)

        # connect the constrolls 
        layer.connectControls()

        # add layer to layer dict
        self.layers[name]=layer



    def setLayerData(self,name,data):
        layer = self.layers[name]
        layer.checkData(data=data)
        layer.setData(data=data)

    def removeLayer(self,name):
        layer = self.layers[name]
        self.layerControl.removeLayerControl(layer)
        layer.cleanUp()
        del self.layers[name]


    def show(self):
        self.win.show()

    def autoRange(self,*args,**kwargs):
        self.layerView.viewBox.autoRange(*args,**kwargs)
