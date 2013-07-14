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

app = QtGui.QApplication([])







class ImageFlowchar(Flowchart):
    def __init__(self,*args,**kwargs):
        print "my flow chart constructor"
        super(ImageFlowchar, self).__init__(*args,**kwargs)
        self.setCurrentFile=self._setCurrentFile


    def _setCurrentFile(self, fileName):
        fileName = str(fileName)
        self.currentFileName = fileName
        if fileName is None:
            self.ui.fileNameLabel.setText("<b>[ new ]</b>")
        else:
            self.ui.fileNameLabel.setText("<b>%s</b>" % os.path.split(str(self.currentFileName))[1])
        self.resizeEvent(None)

    def _fileSaved(self, fileName):
        print "jeah"
        self.setCurrentFile(fileName)
        self.ui.saveBtn.success("Saved.")

## Create main window with a grid layout inside
win = QtGui.QMainWindow()
win.setWindowTitle('pyqtgraph example: FlowchartCustomNode')
cw = QtGui.QWidget()
win.setCentralWidget(cw)
layout = QtGui.QGridLayout()
cw.setLayout(layout)

## Create an empty flowchart with a single input and output
fc = ImageFlowchar(terminals={
    'dataIn': {'io': 'in'},
    'dataOut': {'io': 'out'}    
})
w = fc.widget()

layout.addWidget(fc.widget(), 0, 0, 0, 1)

## Create two ImageView widgets to display the raw and processed data with contrast
## and color control.


# pyqtgraph.functions 


   


v1 = ClickImageView()
v2 = ClickImageView()
v3 = ClickImageView()
v4 = ClickImageView()
layout.addWidget(v1, 0, 1)
layout.addWidget(v2, 0, 2)
layout.addWidget(v3, 1, 1)
layout.addWidget(v4, 1, 2)

win.show()

## generate random input data
#data = np.random.normal(size=(100,100))
#data[40:60, 40:60] += 15.0
#data[30:50, 30:50] += 15.0

data = vigra.readImage('/home/phil/Downloads/lena512.bmp')
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
v1Node.setView(v1)

v2Node = fc.createNode('ImageView', pos=(150, -150))
v2Node.setView(v2)

v3Node = fc.createNode('ImageView', pos=(300, -150))
v3Node.setView(v3)

v3Node = fc.createNode('ImageView', pos=(450, -150))
v3Node.setView(v4)


fNode = fc.createNode('NonLinearDiffusion', pos=(0, 0))
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
