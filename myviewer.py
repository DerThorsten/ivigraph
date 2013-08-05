import numpy as np
import vigra

from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg
import layerviewer as lv











app = QtGui.QApplication([])

viewer =  lv.LayerViewer()
viewer.show()

rgbImg   = vigra.readImage('/home/tbeier/Desktop/lena.bmp')
gradMag  = vigra.filters.gaussianGradientMagnitude(rgbImg,3.5)
seg,nSeg = vigra.analysis.watersheds(gradMag) 


print seg,nSeg


# rgb layer
viewer.addLayer(name='Rgb',layerType='RgbLayer')
viewer.setLayerData(name='Rgb',data=rgbImg)

# gray layer
viewer.addLayer(name='GradMag',layerType='GrayLayer')
viewer.setLayerData(name='GradMag',data=gradMag)


# seg layer
viewer.addLayer(name='SuperPixels',layerType='SegmentationLayer')
viewer.setLayerData(name='SuperPixels',data=seg)


#viewer.addLayer(name='mgray',layerType='MultiGrayLayer')
#viewer.setLayerData(name='mgray',data=rgbImg)


viewer.autoRange()

#viewer.removeLayer(name='rgb-1')
"""

# add layers

viewer.addLayer('data1')
viewer.setLayerData('data1',data1)


data2 = vigra.readImage('/home/tbeier/Desktop/lena.bmp')[:,:,0]
viewer.addLayer('data2')
viewer.setLayerData('data2',data2)

data3 = vigra.readImage('/home/tbeier/Desktop/lena.bmp')[:,:,1]
viewer.addLayer('data3')
viewer.setLayerData('data3',data3)

data4 = vigra.readImage('/home/tbeier/Desktop/lena.bmp')[:,:,2]
viewer.addLayer('data4')
viewer.setLayerData('data4',data4)

#viewer.removeLayer('data2')
"""

QtGui.QApplication.instance().exec_()