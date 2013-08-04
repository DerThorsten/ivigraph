import numpy as np
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg
from collections import OrderedDict
import pyqtgraph.parametertree.parameterTypes as pTypes
from pyqtgraph.parametertree import Parameter, ParameterTree, ParameterItem, registerParameterType
from helpers import setPolicy
from abc import ABCMeta,abstractmethod

from inputCheck import InputCheck
#from lazycall   import lazyCall
from normalize import norm01


class LayerBase(object):
    #__metaclass__ = ABCMeta
    """ Base Class for a Layer of LayerViewer:
        Holds all qt/pyqtgraph items
        All items which are needed to display a certain LayerView
        are stored withn this base class
    """
    def __init__(self,name,layerViewer):
        self.name            = name
        self.viewBox        = layerViewer.layerView.viewBox
        self.layerParameter = None
        self.items          = set()
        self.data           = None
        self.layers         = layerViewer.layers
        self.layerViewer    = layerViewer
        self.layerView      = self.layerViewer.layerView

    def baseControlTemplate(self):
        return [
            {'name': 'ShowLayer', 'type': 'bool', 'value': True, 'tip': "Show / Hide This Layer"},
            {'name': 'HideOthers', 'type': 'action'}
        ]

    def connectParam(self,namePath,f):
        if isinstance(namePath,str):
            namePath=[namePath]
        param =  self.layerParameter.param(*namePath)
        param.sigTreeStateChanged.connect(f)

    def connectBaseControls(self):
        self.connectParam('ShowLayer' ,self.onShowLayerChanged)
        self.connectParam('HideOthers',self.onHideOthers)
        # must be implemented by child classes
        self.connectControls()

    def onHideOthers(self,param,changes):
        #print self.name,"hide others"
        self.layerParameter.param('ShowLayer').setValue(True)
        for otherLayerName in self.layers.keys():
            otherLayer = self.layers[otherLayerName]
            if otherLayer!=self:
                #otherLayer.hideItems()
                otherLayer.layerParameter.param('ShowLayer').setValue(False)


    def onShowLayerChanged(self,param,changes):
        assert len(changes)==1
        param,change,show = changes[0]
        if show == True :
            self.showItems()
        else:
            self.hideItems()

    def setLayerParameter(self,parameter):
        self.layerParameter=parameter

    def layerName(self):
        return self.name

    def getParamValue(self,*args):
        return self.layerParameter.param(*args).value()

    def addItem(self,item):
        self.items.add(item)
        self.viewBox.addItem(item)

    def removeItem(self,item):
        assert item in self.items
        self.viewBox.removeItem(item)
        self.items.remove(item)

    def removeItems(self):
        for item in self.items:
            self.viewBox.removeItem(item)
        self.items.clear()

    def cleanUp(self):
        self.removeItems()

    def hideItems(self):
        for item in self.items:
            item.hide()

    def showItems(self):
        for item in self.items:
            # print item
            item.show()
            #item.update()

 
    def checkData(self,data):
        raise NotImplementedError("Needs to be implemted by subclas")

    def setData(self,data):
        raise NotImplementedError("Needs to be implemted by subclas")

    def controlTemplate(self):
        raise NotImplementedError("Needs to be implemted by subclas")

    def connectControls(self):
        raise NotImplementedError("Needs to be implemted by subclas")

    def lala(self):
        raise NotImplementedError("Needs to be implemted by subclas")


layerTypes=dict()

class LayerImageItem(pg.ImageItem):
    def __init__(self,*args,**kwargs):
        super(LayerImageItem,self).__init__(*args,**kwargs)

    def mouseClickEvent(self, ev):
        print type(self)
        print ev.pos()


class ImageRgbLayer(LayerImageItem,LayerBase):

    def __init__(self,name,layerViewer):
        LayerBase.__init__(self,name,layerViewer)
        LayerImageItem.__init__(self,parent=self.layerView)
        self.addItem(self)

    def checkData(self,data):
        InputCheck.colorImage(data)

    def setData(self,data=None):
        if data is None:
            self.removeItems()
            return 
        else:
            self.data=data
            self.setImage(self.data,opacity=self.getParamValue('Opacity'))

    def controlTemplate(self):
        return [{'name': 'Opacity', 'type': 'float', 'value': 0.75, 'step': 0.1,'limits':[0,1]} ] 

    def connectControls(self):

        def onOpacityChanged(param,changes):
            assert len(changes)==1
            param,change,opacity = changes[0]
            self.setImage(self.data,opacity=opacity)

        self.connectParam('Opacity',onOpacityChanged)



class ImageGrayLayer(LayerImageItem,LayerBase):


    def __init__(self,name,layerViewer):
        LayerBase.__init__(self,name,layerViewer)
        LayerImageItem.__init__(self,parent=self.layerView)
        # add item to item(set)
        self.addItem(self)
        self.preProcessedData = None
        self.cmappedData      = None
        #self.arrowItem = pg.ArrowItem(pos=(50.5,90),parent=self.imageItem,pxMode=False)
        #self.addItem(self.arrowItem)
        #self.arrowItem.hide()
        #self.arrowItem.show()

    def checkData(self,data):
        InputCheck.grayImage(data)

    def setData(self,data=None):
        if data is None:
            self.removeItems()
            return 
        else:
            self.data=data
            self.preProcessedData   = norm01(np.squeeze(self.data))
            self.cmappedData        = self.getParamValue('ColorMap').map(self.preProcessedData)
            self.setImage(self.cmappedData,opacity=self.getParamValue('Opacity'))

    def controlTemplate(self):
        return [
            {'name': 'ColorMap', 'type': 'colormap'},
            {'name': 'Opacity', 'type': 'float', 'value': 0.75, 'step': 0.1,'limits':[0,1]} 
        ] 

    def connectControls(self):

        def onOpacityChanged(param,changes):
            assert len(changes)==1
            param,change,opacity = changes[0]
            self.setImage(self.cmappedData,opacity=opacity)

        def onColorMapChanged(param,changes):
            assert len(changes)==1
            param,change,cmap = changes[0]
            self.cmappedData  = cmap.map(self.preProcessedData)
            self.setImage(self.cmappedData,opacity=self.getParamValue('Opacity'))

        self.connectParam('Opacity',onOpacityChanged)
        self.connectParam('ColorMap',onColorMapChanged)



class ImageMultiGrayLayer(LayerImageItem,LayerBase):


    def __init__(self,name,layerViewer):
        LayerBase.__init__(self,name,layerViewer)
        LayerImageItem.__init__(self,parent=self.layerView)

        self.addItem(self)
        self.preProcessedData   = None
        self.selectedChannelImg = None
        self.cmappedData         = None

    def checkData(self,data):
        InputCheck.multiGrayImage(data)

    def setData(self,data=None):
        if data is None:
            self.removeItems()
            return 
        else:
            self.layerParameter.param('Channel').setOpts(name = 'Channel',value =0 ,limits=[0,data.shape[2]-1])
            self.data               = data
            self.preProcessedData   = norm01(self.data,channelWise=True)
            self.selectedChannelImg = self.preProcessedData[:,:,self.getParamValue('Channel')]
            self.cmappedData        = self.getParamValue('ColorMap').map(self.selectedChannelImg)
            self.setImage(self.cmappedData,opacity=self.getParamValue('Opacity'))
    

    def controlTemplate(self):
        return [
            {'name': 'Channel', 'type': 'int','value':0},
            {'name': 'ColorMap', 'type': 'colormap'},
            {'name': 'Opacity', 'type': 'float', 'value': 0.75, 'step': 0.1,'limits':[0,1]} 
        ] 

    def connectControls(self):

        def onChannelChanged(param,changes):
            if self.data is None :
                return
            channel                 = self.getParamValue('Channel')
            self.selectedChannelImg = self.preProcessedData[:,:,channel]
            self.cmappedData        = self.getParamValue('ColorMap').map(self.selectedChannelImg)
            self.setImage(self.cmappedData,opacity=self.getParamValue('Opacity'))
    
        def onColorMapChanged(param,changes):
            assert len(changes)==1
            param,change,cmap = changes[0]
            self.cmappedData  = cmap.map(self.selectedChannelImg)
            self.setImage(self.cmappedData,opacity=self.getParamValue('Opacity'))

        def onOpacityChanged(param,changes):
            assert len(changes)==1
            param,change,opacity = changes[0]
            self.setImage(self.cmappedData,opacity=opacity)

        self.connectParam('Channel',onChannelChanged)
        self.connectParam('Opacity',onOpacityChanged)
        self.connectParam('ColorMap',onColorMapChanged)


layerTypes['RgbLayer']=ImageRgbLayer
layerTypes['GrayLayer']=ImageGrayLayer
layerTypes['MultiGrayLayer']=ImageMultiGrayLayer