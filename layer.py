
from pyqtgraph.flowchart import Flowchart, Node
import pyqtgraph.flowchart.library as fclib
from pyqtgraph.flowchart.library.common import CtrlNode
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg
import numpy as np
import numpy
import scipy.ndimage
import vigra

from layerimageview import LayerImageView

from views import * 
from operators import * 

app = QtGui.QApplication([])




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

gradMag = numpy.squeeze(vigra.filters.gaussianGradientMagnitude(data,2.0))


viewer = LayerImageView()
viewer.show()

viewer.addLayer('main-image')

viewer.setImage(gradMag)

if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
