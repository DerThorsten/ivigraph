import numpy as np
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg



def setPolicy(widget,p=QtGui.QSizePolicy.Maximum):
    sizePolicy = QtGui.QSizePolicy(p,p)
    sizePolicy.setHorizontalStretch(0)
    sizePolicy.setVerticalStretch(0)
    widget.setSizePolicy(sizePolicy)