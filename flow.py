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
    batchMode=False
if hname=='tbeier-A780GM-A':
    data = vigra.readImage('/home/tbeier/Desktop/lena.bmp')
    dataFolder = "/home/tbeier/src/privatOpengm/experiments/datasets/bsd500/BSR/BSDS500/data/images/val/"
    fFilter='.jpg'
    batchMode=True
if hname =='tbeier-vaio':
   data = vigra.readImage('/home/tbeier/Desktop/lena.jpg')
   batchMode=False


ivigraph = IViGrahp()
ivigraph.win.show()
if batchMode :
    ivigraph.setBatchInput(folder=dataFolder,fFilter=fFilter)
else :
    ivigraph.setInput(dataIn=data)
fc=ivigraph.flowChart
viewerNodes=ivigraph.viewerNodes
# custom connections
fNode = fc.createNode('TestNode', pos=(0, 0))
fc.connectTerminals(fc['dataIn'], fNode['dataIn'])
fc.connectTerminals(fc['dataIn'], viewerNodes[0]['data'])
#fc.connectTerminals(v1Node['view'], fNode['view'])
fc.connectTerminals(fNode['dataOut'], viewerNodes[1]['data'])
fc.connectTerminals(fNode['dataOut'], fc['dataOut'])



QtGui.QApplication.instance().exec_()