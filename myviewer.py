import numpy as np
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg
from collections import OrderedDict


class Layer(object):

    class Ui(object):
        def __init__(self):
            self.layout=None

    def __init__(self,name,imageItem=None,inputData=None,processedData=None):
        self.name=name
        if imageItem is not None:
            self.imageItem=imageItem
        else :
            self.imageItem=pg.ImageItem(border='r')
        self.inputData=inputData
        self.processedData=processedData

        self.ui = Layer.Ui()

    def setImage(self,data):
        self.inputData=data
        self.imageItem.setImage(data)


class LayerdViewer(QtGui.QWidget):

    def setuptMainLayouts(self):
        # ui widgets
        self.mainVLayout = QtGui.QVBoxLayout(self)
        self.viewAndRightLayout= QtGui.QHBoxLayout()
        self.belowViewLayout= QtGui.QHBoxLayout()
        
        self.mainVLayout.addLayout(self.viewAndRightLayout)
        self.mainVLayout.addLayout(self.belowViewLayout)
        
        
        

    def setupBelowViewUi(self):
        toyBt1 = QtGui.QPushButton("testBelow1")
        toyBt2 = QtGui.QPushButton("testBelow2")
        self.belowViewLayout.addWidget(toyBt1)
        self.belowViewLayout.addWidget(toyBt2)
        self.belowViewLayout.addStretch(1)

    def setupRightToViewUi(self):
        self.rightToViewLayout=QtGui.QVBoxLayout()
        self.viewAndRightLayout.addLayout(self.rightToViewLayout)
        toyBt1 = QtGui.QPushButton("testRight1")
        toyBt2 = QtGui.QPushButton("testRight2")
        self.rightToViewLayout.addWidget(toyBt1)
        self.rightToViewLayout.addWidget(toyBt2)
        self.rightToViewLayout.addStretch(1)

    def setupViewBoxUi(self):
        self.win = QtGui.QMainWindow()
        self.win.setWindowTitle("Layer")
        self.win.setCentralWidget(self)
        #self.win.resize(1000,500)

        # graph view and layout of gaphview
        self.graphView       = pg.GraphicsView()
        self.graphViewLayout = QtGui.QGraphicsGridLayout()
        self.graphView.centralWidget.setLayout(self.graphViewLayout)

        
        self.viewAndRightLayout.addWidget(self.graphView)

        # view box
        self.viewBox = pg.ViewBox()
        self.viewBox.setAspectLocked(True)

        # add view box to graph view layout
        self.graphViewLayout.addItem(self.viewBox,0,0)

    def __init__(self,parent=None):


        QtGui.QWidget.__init__(self, parent)
        self.setuptMainLayouts()
        self.setupViewBoxUi()
        self.setupBelowViewUi()
        self.setupRightToViewUi()

        self.layers = OrderedDict()


    def addLayer(self,name):
        layer = Layer(name=name)
        self.layers[name]=layer
        self.viewBox.addItem(layer.imageItem)
        self.addLayerGui(name)

    def addLayerGui(self,name):
        layer = self.layers[name]
        layout = layer.ui.layout
        layout = QtGui.QHBoxLayout()
        self.rightToViewLayout.addLayout(layout)
        
        toyBt1 = QtGui.QPushButton(layer.name+'1')
        toyBt2 = QtGui.QPushButton(layer.name+'2')
        toyBt3 = QtGui.QPushButton(layer.name+'3')
        layout.addWidget(toyBt1)
        layout.addWidget(toyBt2)
        layout.addWidget(toyBt3)
        layer.ui.layout=layout

    def removeLayer(self,name):
        layer = self.layers[name]
        self.viewBox.removeItem(layer.imageItem)
        self.removeLayerGui(name)
        del self.layers[name]

    def removeLayerGui(self,name):
        layer = self.layers[name]
        layoutToRemove = layer.ui.layout
        for i in range(0, layoutToRemove.count()):
            layoutToRemove.itemAt(i).widget().hide()
        self.rightToViewLayout.removeItem(layoutToRemove)

   

    def setLayerData(self,name,data):
        layer = self.layers[name]
        layer.setImage(data)


    def show(self):
        self.win.show()





app = QtGui.QApplication([])


viewer =  LayerdViewer()
viewer.show()



# add layers
data1 = np.random.normal(size=(600, 300), loc=1024, scale=64).astype(np.float32)
viewer.addLayer('data1')
viewer.setLayerData('data1',data1)


data2 = np.random.normal(size=(600, 200), loc=1024, scale=64).astype(np.float32)
viewer.addLayer('data2')
viewer.setLayerData('data2',data2)

viewer.removeLayer('data2')


QtGui.QApplication.instance().exec_()