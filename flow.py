# -*- coding: utf-8 -*-
"""
This example demonstrates writing a custom Node subclass for use with flowcharts.

We implement a couple of simple image processing nodes.
"""


from pyqtgraph.flowchart import Flowchart, Node,FlowchartCtrlWidget
import pyqtgraph.flowchart.library as fclib
from pyqtgraph.flowchart.library.common import CtrlNode
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg
import numpy as np
import numpy
import scipy.ndimage
import vigra

from views import * 
from operators import * 
from pyqtgraph.dockarea import *

app = QtGui.QApplication([])


import types



## Create main window with a grid layout inside\
#app = QtGui.QApplication([])
win = QtGui.QMainWindow()
win.setWindowTitle('pyqtgraph example: FlowchartCustomNode')
area = DockArea()
win.setCentralWidget(area)
win.resize(1000,500)
win.setWindowTitle('pyqtgraph example: dockarea')







## Create an empty flowchart with a single input and output
fc = Flowchart(terminals={
    'dataIn': {'io': 'in'},
    'dataOut': {'io': 'out'}    
})
w = fc.widget()









#layout.addWidget(fc.widget(), 0, 0, 0, 1)


d1  = Dock("Controll", size=(1, 1))

viewDocks = [   Dock("view0", size=(1, 1)), Dock("view1", size=(1, 1)),
                Dock("view2", size=(1, 1)), Dock("view3", size=(1, 1)) ]

area.addDock(d1, 'left')
area.addDock(viewDocks[0], 'right')
area.addDock(viewDocks[3], 'below', viewDocks[0])
area.addDock(viewDocks[2], 'below', viewDocks[0])
area.addDock(viewDocks[1], 'below', viewDocks[0])

viewers = [ ClickImageView(),ClickImageView(),
            ClickImageView(),ClickImageView()]


d1.addWidget(fc.widget())
for viewDock,viewer in zip(viewDocks,viewers):
    viewDock.addWidget(viewer)


win.show()

## generate random input data
#data = np.random.normal(size=(100,100))
#data[40:60, 40:60] += 15.0
#data[30:50, 30:50] += 15.0

import socket
hname = socket.gethostname()

print "host name ",hname

if hname=='Beethoven':
    data = vigra.readImage('/home/phil/Downloads/lena512.bmp')
if hname=='tbeier-A780GM-A':
    data = vigra.readImage('/home/tbeier/Desktop/lena.bmp')
if hname =='tbeier-vaio':
   data = vigra.readImage('/home/tbeier/Desktop/lena.jpg')
data = data[:,:,:]
#data = numpy.squeeze(data[:,:,0])
vigra.filters.discMedian(data.astype(numpy.uint8),1)
#data += np.sin(np.linspace(0, 100, 1000))
#data = metaarray.MetaArray(data, info=[{'name': 'Time', 'values': np.linspace(0, 1.0, len(data))}, {}])

## Set the raw data as the input value to the flowchart
fc.setInput(dataIn=data)


## Now we will programmatically add nodes to define the function of the flowchart.
## Normally, the user will do this manually or by loading a pre-generated
## flowchart file.

v1Node = fc.createNode('ImageView', pos=(0, -150))
v1Node.setView(viewers[0])

v2Node = fc.createNode('ImageView', pos=(150, -150))
v2Node.setView(viewers[1])

v3Node = fc.createNode('ImageView', pos=(300, -150))
v3Node.setView(viewers[2])

v4Node = fc.createNode('ImageView', pos=(450, -150))
v4Node.setView(viewers[3])


"""
The following will fix save and loading

"""

import os
def setCurrentFileFixed(self, fileName):
    self.currentFileName = fileName
    if fileName is None:
        self.ui.fileNameLabel.setText("<b>[ new ]</b>")
    else:
        self.ui.fileNameLabel.setText("<b>%s</b>" % os.path.split(str(self.currentFileName))[1])
    self.resizeEvent(None)
FlowchartCtrlWidget.setCurrentFile=setCurrentFileFixed
def loadFile(self, fileName=None, startDir=None,nodes=(v1Node,v2Node,v3Node,v4Node),viewers=viewers):
    import pyqtgraph.configfile as configfile
    if fileName is None:
        if startDir is None:
            startDir = self.filePath
        if startDir is None:
            startDir = '.'
        self.fileDialog = pg.FileDialog(None, "Load Flowchart..", startDir, "Flowchart (*.fc)")
        #self.fileDialog.setFileMode(QtGui.QFileDialog.AnyFile)
        #self.fileDialog.setAcceptMode(QtGui.QFileDialog.AcceptSave) 
        self.fileDialog.show()
        self.fileDialog.fileSelected.connect(self.loadFile)
        return
        ## NOTE: was previously using a real widget for the file dialog's parent, but this caused weird mouse event bugs..
        #fileName = QtGui.QFileDialog.getOpenFileName(None, "Load Flowchart..", startDir, "Flowchart (*.fc)")
    fileName = str(fileName)
    state = configfile.readConfigFile(fileName)
    self.restoreState(state, clear=True)
    self.viewBox.autoRange()
    #self.emit(QtCore.SIGNAL('fileLoaded'), fileName)
    self.sigFileLoaded.emit(fileName)

    for name, node in self._nodes.items():

        print name
        if isinstance(node,v1Node.__class__):
            #node.view.updateNorm()
            #node.view.normRadioChanged()
            #node.update()
            #node.update()
            if name =='ImageView.0':
                node.setView(viewers[0])
            if name =='ImageView.1':
                node.setView(viewers[1])
            if name =='ImageView.2':
                node.setView(viewers[2])
            if name =='ImageView.3':
                node.setView(viewers[3])

    self.inputNode.update()

    """
    for name, node in self._nodes.items():
        print name
        if isinstance(node,self.__class__):
            print "bingo node...."
        node.update()

    assert False
    """
    """
    for name, node in self._nodes.items():

        print name
        if isinstance(node,v1Node.__class__):
            node.view.updateNorm()
            node.view.normRadioChanged()
            node.view.ui.histogram.update()
            node.view.ui.histogram.imageChanged()
            node.update()
    """
    #for node,viewer in zip(nodes,viewers):
    #    node.setView(viewer)
fc.loadFile=types.MethodType(loadFile,fc)

fNode = fc.createNode('GaussianGradientMagnitude', pos=(0, 0))
fc.connectTerminals(fc['dataIn'], fNode['dataIn'])
fc.connectTerminals(fc['dataIn'], v1Node['data'])
#fc.connectTerminals(v1Node['view'], fNode['view'])
fc.connectTerminals(fNode['dataOut'], v2Node['data'])
fc.connectTerminals(fNode['dataOut'], fc['dataOut'])


## Start Qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
