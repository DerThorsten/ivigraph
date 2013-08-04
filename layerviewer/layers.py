import numpy as np
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg
from collections import OrderedDict
import pyqtgraph.parametertree.parameterTypes as pTypes
from pyqtgraph.parametertree import Parameter, ParameterTree, ParameterItem, registerParameterType
from helpers import setPolicy
from abc import ABCMeta,abstractmethod

from inputCheck import InputCheck

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

    def connectBaseControls(self):
        param = self.layerParameter.param('ShowLayer')
        param.sigTreeStateChanged.connect(self.onShowLayerChanged)

        param = self.layerParameter.param('HideOthers')
        param.sigTreeStateChanged.connect(self.onHideOthers)

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

        print self.name,"show",show
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
            print item
            item.show()
            item.update()

 
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

class ImageRgbLayer(LayerBase,LayerImageItem):


    def __init__(self,name,layerViewer):

        LayerBase.__init__(self,name,layerViewer)
        LayerImageItem.__init__(self,parent=self.layerView)


        #self.imageItem = pg.ImageItem(parent=self.layerView)


        # add item to item(set)
        self.addItem(self)
        self.preProcessedData = None

    def mouseClickEvent(self, ev):
        print type(self)
        print ev.pos()
       
    def checkData(self,data):
        InputCheck.colorImage(data)

    def setData(self,data=None):
        if data is None:
            self.removeItems()
            return 
        else:
            self.data=data
            self.preProcessData()
            self.showData()

    def preProcessData(self):
        """ rgb layer needs no preprocessing ?!?"""
        self.preProcessedData=self.data

    def showData(self,opacity=None):
        """ this will make the data visible
        """
        if opacity is None:
            opacity=self.getParamValue('Opacity')

        self.setImage(self.preProcessedData,opacity=opacity)
        #self.imageItem.update()
        #self.viewBox.update()

    def controlTemplate(self):
        return [{'name': 'Opacity', 'type': 'float', 'value': 0.75, 'step': 0.1,'limits':[0,1]} ] 

    def connectControls(self):
        opacityParam = self.layerParameter.param('Opacity')
        opacityParam.sigTreeStateChanged.connect(self.onOpacityChanged)


    def onOpacityChanged(self,param,changes):
        assert len(changes)==1
        param,change,opacity = changes[0]
        self.showData(opacity=opacity)

class ImageGrayLayer(LayerBase,LayerImageItem):


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


        #Wprint "numitems",len(self.items)
    def checkData(self,data):
        error = RuntimeError("ImageGrayLayer data must have 2 dimensions or 3 dimensions with shape[2]=1")
        if data is None:
            return
        if data.ndim==3:
            if data.shape[2]!=1:
                raise error
            return
        if data.ndim!=2 :
            raise error


    def setData(self,data=None):
        if data is None:
            self.removeItems()
            return 
        else:
            self.data=data
            self.preProcessData()
            self.doColorMapping()
            self.showData()

    def doColorMapping(self,cmap=None):
        if cmap is None:
            cmap =self.getParamValue('ColorMap')

        self.cmappedData  = cmap.map(self.preProcessedData)

    def preProcessData(self):
        """ rgb layer needs no preprocessing ?!?"""
        self.preProcessedData=self.data.copy()
        self.preProcessedData=np.squeeze(self.preProcessedData)
        self.preProcessedData-=self.preProcessedData.min()
        self.preProcessedData/=self.preProcessedData.max()

    def showData(self,opacity=None):
        """ this will make the data visible
        """
        if opacity is None:
            opacity=self.getParamValue('Opacity')
    
        self.setImage(self.cmappedData,opacity=opacity)
        #self.imageItem.update()
        #self.viewBox.update()

    def controlTemplate(self):
        return [
            {'name': 'ColorMap', 'type': 'colormap'},
            {'name': 'Opacity', 'type': 'float', 'value': 0.75, 'step': 0.1,'limits':[0,1]} 
        ] 

    def connectControls(self):
        param = self.layerParameter.param('Opacity')
        param.sigTreeStateChanged.connect(self.onOpacityChanged)

        param = self.layerParameter.param('ColorMap')
        param.sigTreeStateChanged.connect(self.onColorMapChanged)

    def onOpacityChanged(self,param,changes):
        assert len(changes)==1
        #print "arrorw..."
        param,change,opacity = changes[0]
        #self.arrowItem.setStyle(headLen=int(5*opacity)+1,
        #   tipAngle=0, baseAngle=15, tailLen=10, tailWidth=3,pxMode=False
        #)
        self.showData(opacity=opacity)

    def onColorMapChanged(self,param,changes):
        assert len(changes)==1
        param,change,cmap = changes[0]
        self.doColorMapping(cmap=cmap)
        self.showData()

class ImageMultiGrayLayer(LayerBase,LayerImageItem):


    def __init__(self,name,layerViewer):
        LayerBase.__init__(self,name,layerViewer)
        LayerImageItem.__init__(self,parent=self.layerView)


        # add item to item(set)
        self.addItem(self)
        


        self.preProcessedData         = None
        self.selectedChannelImg      = None
        self.cmappedData              = None

    def checkData(self,data):
        error = RuntimeError("ImageMultiGrayLayer data must have 3 dimensions")
        if data is None:
            return
        if data.ndim!=3:
            raise error



    def setData(self,data=None):
        if data is None:
            self.removeItems()
            return 
        else:
            self.data=data
            self.preProcessData()
            nChannels = self.data.shape[2]
            self.selectedChannelImg=self.preProcessedData[:,:,self.getParamValue('Channel')]
            self.layerParameter.param('Channel').setOpts(
                name = 'Channel',
                value =0 ,
                limits=[0,nChannels-1]
            )
            
            
            self.doColorMapping()
            self.showData()

    def doColorMapping(self,cmap=None):
        if cmap is None:
            cmap =self.getParamValue('ColorMap')
        self.cmappedData  = cmap.map(self.selectedChannelImg)


    def preProcessData(self):
        self.preProcessedData=self.data.copy()
        for c in range(self.preProcessedData.shape[2]):
            self.preProcessedData[:,:,c]-=self.preProcessedData[:,:,c].min()
            self.preProcessedData[:,:,c]/=self.preProcessedData[:,:,c].max()

    def showData(self,opacity=None):
        """ this will make the data visible
        """
        if opacity is None:
            opacity=self.getParamValue('Opacity')
    
        self.setImage(self.cmappedData,opacity=opacity)
        #self.imageItem.update()
        #self.viewBox.update()

    def controlTemplate(self):
        return [
            {'name': 'Channel', 'type': 'int','value':0},
            {'name': 'ColorMap', 'type': 'colormap'},
            {'name': 'Opacity', 'type': 'float', 'value': 0.75, 'step': 0.1,'limits':[0,1]} 
        ] 

    def connectControls(self):
        param = self.layerParameter.param('Channel')
        param.sigTreeStateChanged.connect(self.onChannelChanged)

        param = self.layerParameter.param('ColorMap')
        param.sigTreeStateChanged.connect(self.onColorMapChanged)

        param = self.layerParameter.param('Opacity')
        param.sigTreeStateChanged.connect(self.onOpacityChanged)



    def onChannelChanged(self,param,changes):
        if self.data is None:
            return 
        channel = self.getParamValue('Channel')
        self.selectedChannelImg=self.preProcessedData[:,:,channel]

        self.doColorMapping()
        self.showData()

    
    def onColorMapChanged(self,param,changes):
        assert len(changes)==1
        param,change,cmap = changes[0]
        self.doColorMapping(cmap=cmap)
        self.showData()

    def onOpacityChanged(self,param,changes):
        assert len(changes)==1
        param,change,opacity = changes[0]
        self.showData(opacity=opacity)



layerTypes['RgbLayer']=ImageRgbLayer
layerTypes['GrayLayer']=ImageGrayLayer
layerTypes['MultiGrayLayer']=ImageMultiGrayLayer