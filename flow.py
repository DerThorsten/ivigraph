import socket
import numpy as np
import vigra

import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore
from ivigraph import IViGrahp


hname = socket.gethostname()



app = QtGui.QApplication([])


print "host name ",hname
if hname=='Beethoven':
    data = vigra.readImage('/home/phil/Downloads/lena512.bmp')
if hname=='tbeier-A780GM-A':
    data = vigra.readImage('/home/tbeier/Desktop/lena.bmp')
if hname =='tbeier-vaio':
   data = vigra.readImage('/home/tbeier/Desktop/lena.jpg')



ivigraph = IViGrahp()
ivigraph.win.show()
ivigraph.setInput(dataIn=data)

fc=ivigraph.flowChart
viewerNodes=ivigraph.viewerNodes
# custom connections
fNode = fc.createNode('SlicSuperpixels', pos=(0, 0))
fc.connectTerminals(fc['dataIn'], fNode['dataIn'])
fc.connectTerminals(fc['dataIn'], viewerNodes[0]['data'])
#fc.connectTerminals(v1Node['view'], fNode['view'])
fc.connectTerminals(fNode['dataOut'], viewerNodes[1]['data'])
fc.connectTerminals(fNode['dataOut'], fc['dataOut'])



QtGui.QApplication.instance().exec_()