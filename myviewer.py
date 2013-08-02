import numpy as np
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg
from collections import OrderedDict
import vigra




class LayerBase(QtGui.QWidget):
    def __init__(self):
        pass


    def inputToRGBA(**kwargs):
        pass




class LayerBase(object):
    pass

class ImageLayerBase(LayerBase):
    pass


class RgbLayer(ImageLayerBase):
    pass

class MultiRgbLayer(ImageLayerBase):
    pass

class GrayLayer(ImageLayerBase):
    pass

class MultiGrayLayer(ImageLayerBase):
    pass













sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding,
                                         QtGui.QSizePolicy.Expanding)
sizePolicy.setHorizontalStretch(0)
sizePolicy.setVerticalStretch(0)


def setExpanding(widget):
    sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding,
                                         QtGui.QSizePolicy.Expanding)
    sizePolicy.setHorizontalStretch(0)
    sizePolicy.setVerticalStretch(0)

    widget.setSizePolicy(sizePolicy)

def setPolicy(widget,p=QtGui.QSizePolicy.Maximum):
    
    sizePolicy = QtGui.QSizePolicy(p,p)
    sizePolicy.setHorizontalStretch(0)
    sizePolicy.setVerticalStretch(0)

    widget.setSizePolicy(sizePolicy)

class Layer(object):

    class Ui(object):
        def __init__(self,parent):
            self.parent   = parent
            self.layout   = QtGui.QVBoxLayout()
            self.layoutR1 = QtGui.QHBoxLayout()
            self.layoutR2 = QtGui.QHBoxLayout()
            self.layoutR3 = QtGui.QHBoxLayout()

            
            #self.layout.addWidget(self.nameLabel)
            self.layout.addLayout(self.layoutR1)
            self.layout.addLayout(self.layoutR2)
            #self.layout.addLayout(self.layoutR3)

            self.setupLayoutR1()
            self.setupLayoutR2()
            #self.setupLayoutR3()

        def setupLayoutR1(self):
            name = str(self.parent.layerdViewer.numLayers())
            self.nameLabel = QtGui.QLabel(name)
            self.comboBoxImageType = QtGui.QComboBox()
            self.comboBoxImageType.addItems(['Rgb','Lab','C'])
            self.channelSliderChannelDesc=QtGui.QLabel("C:")
            self.sliderChannels = QtGui.QSlider(QtCore.Qt.Horizontal)

            self.sliderChannels.setMinimumSize(30,10)
            setPolicy(self.sliderChannels,QtGui.QSizePolicy.Maximum)


            self.sliderChannels.setMinimum(0)
            self.sliderChannels.setMaximum(10)
            self.sliderChannels.setTickInterval(1)
            self.sliderChannels.setSingleStep(1)
            self.sliderChannels.setValue(0)
            self.sliderChannels.setTickPosition(QtGui.QSlider.TicksAbove)
            self.labelCurrentChannel=QtGui.QLabel(str(self.sliderChannels.sliderPosition()))

            self.layoutR1.addWidget(self.nameLabel)
            self.layoutR1.addWidget(self.comboBoxImageType)
            self.layoutR1.addWidget(self.channelSliderChannelDesc)
            self.layoutR1.addWidget(self.sliderChannels)
            self.layoutR1.addWidget(self.labelCurrentChannel)
            #self.layoutR1.addStretch(1)
        def setupLayoutR2(self):

            #self.labelColormapDesc=QtGui.QLabel("CMap:")
            self.colormap  = pg.GradientWidget()
            #self.colormap.setMinimumSize(50,10)
            self.colormap.setMaximumSize(70,7)
            setPolicy(self.colormap,QtGui.QSizePolicy.Maximum)
            self.colormap.setLength(5)
            self.colormap.update()
            #self.layoutR2.addWidget(self.labelColormapDesc)
            self.layoutR2.addWidget(self.colormap)
            #self.layoutR2.addStretch(1)

        #def setupLayoutR3(self):
            self.labelAlphaDesc=QtGui.QLabel("A:")
            self.sliderAlpha = QtGui.QSlider(QtCore.Qt.Horizontal)
            setPolicy(self.sliderAlpha,QtGui.QSizePolicy.Maximum)
            self.sliderAlpha.setMinimumSize(30,10)
            self.sliderAlpha.setMinimum(0)
            self.sliderAlpha.setMaximum(100)
            self.sliderAlpha.setTickInterval(1)
            self.sliderAlpha.setSingleStep(1)
            self.sliderAlpha.setValue(50)
            self.labelAlpha=QtGui.QLabel(str(self.sliderAlpha.sliderPosition()/100.0))



            self.layoutR2.addWidget(self.labelAlphaDesc)
            self.layoutR2.addWidget(self.sliderAlpha)
            self.layoutR2.addWidget(self.labelAlpha)
            #self.layoutR3.addStretch(1)
            # connections
            self.sliderAlpha.valueChanged.connect(self.parent.onAlphaChanged) 

     
        def initialzie(self):
            if self.parent.layerdViewer.numLayers()<=0:
                self.parent.onAlphaChanged(100)
            else :
                self.parent.onAlphaChanged(25)

    def onAlphaChanged(self,value):
        self.ui.sliderAlpha.setValue(value)
        self.alpha=float(value)/100.0
        self.ui.labelAlpha.setText("%.2f" % self.alpha)

        print "call processInputData (alpha change)"
        self.processInputData()


    def __init__(self,name,layerdViewer,imageItem=None):

        # name of the layer 
        self.layerdViewer   = layerdViewer
        self.name           = name
        # image item (item will be added to external view box)
        self.imageItem=imageItem
        if imageItem is None:
            self.imageItem      = pg.ImageItem(border='r')

        # attributes related to inout data
        # (will be set from self.setImage )
        self.inputData      = None
        self.processedData  = None
        self.nChannels      = None

        # attributes which will be actualized from ui
        self.alpha          = None
        self.ui             = Layer.Ui(self)
        self.ui.initialzie()
        print "alpha after ui setup",self.alpha

    def processInputData(self):
        if self.inputData is None:
            return

        self.imageItem.setImage(self.inputData,opacity=self.alpha)


    def setImage(self,data):
        self.inputData=data
        self.processInputData()


class LayerdViewer(QtGui.QWidget):

    def setuptMainLayouts(self):
        # ui widgets
        self.mainVLayout            = QtGui.QVBoxLayout(self)
        self.viewAndRightLayout     = QtGui.QHBoxLayout()
        self.belowViewLayout        = QtGui.QHBoxLayout()
        


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
        #toyBt1 = QtGui.QPushButton("testRight1")
        #toyBt2 = QtGui.QPushButton("testRight2")
        #self.rightToViewLayout.addWidget(toyBt1)
        #self.rightToViewLayout.addWidget(toyBt2)
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

        setPolicy(self.graphView,QtGui.QSizePolicy.Expanding)


        self.graphView.setSizePolicy(sizePolicy)

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

    def numLayers(self):
        print "nl ",len(self.layers)
        return len(self.layers)

    def addLayer(self,name):
        layer = Layer(name=name,layerdViewer=self)
        self.layers[name]=layer
        self.viewBox.addItem(layer.imageItem)
        self.addLayerGui(name)

    def addLayerGui(self,name):
        layer = self.layers[name]
        self.rightToViewLayout.addLayout(layer.ui.layout)

    def removeLayer(self,name):
        layer = self.layers[name]
        self.removeLayerGui(name)
        self.viewBox.removeItem(layer.imageItem)
        del self.layers[name]

    def removeLayerGui(self,name):
        layer = self.layers[name]
        self.delLayout(layer.ui.layoutR1,layer.ui.layout)
        self.delLayout(layer.ui.layoutR2,layer.ui.layout)
        self.delLayout(layer.ui.layoutR3,layer.ui.layout)
        self.delLayout(layer.ui.layout,self.rightToViewLayout)

    def delLayout(self,layoutToRemove,parent):
        for i in range(0, layoutToRemove.count()):
            layoutToRemove.itemAt(i).widget().hide()
        parent.removeItem(layoutToRemove)

    def setLayerData(self,name,data):
        flippedData=np.fliplr(data)
        layer = self.layers[name]
        layer.setImage(flippedData)


    def show(self):
        self.win.show()

    def update(self):
        assert False





app = QtGui.QApplication([])


viewer =  LayerdViewer()
viewer.show()



# add layers
data1 = vigra.readImage('/home/tbeier/Desktop/lena.bmp')[:,:,:]
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


QtGui.QApplication.instance().exec_()