import numpy as np
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg



def setPolicy(widget,p=QtGui.QSizePolicy.Maximum):
    sizePolicy = QtGui.QSizePolicy(p,p)
    sizePolicy.setHorizontalStretch(0)
    sizePolicy.setVerticalStretch(0)
    widget.setSizePolicy(sizePolicy)




def permuteLabels(data):

    flat = np.array(data).reshape([-1])
    unique , relabeling = np.unique(flat,return_inverse=True)
    permUnique = np.random.permutation(unique)
    flatNew = permUnique[relabeling]
    newLabels = flatNew.reshape([data.shape[0],data.shape[1]])
    return newLabels