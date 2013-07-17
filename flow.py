# -*- coding: utf-8 -*-
"""
This example demonstrates writing a custom Node subclass for use with flowcharts.

We implement a couple of simple image processing nodes.
"""


from pyqtgraph.flowchart import Flowchart, Node
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

viewDocks = [   Dock("view1", size=(1, 1)), Dock("view2", size=(1, 1)),
                Dock("view3", size=(1, 1)), Dock("view4", size=(1, 1)) ]

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
