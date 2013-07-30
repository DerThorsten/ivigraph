from pyqtgraph.flowchart import Flowchart, Node,FlowchartCtrlWidget
import pyqtgraph.flowchart.library as fclib
from pyqtgraph.flowchart.library.common import CtrlNode
import pyqtgraph.configfile as configfile
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg
import numpy as np
import scipy.ndimage
import vigra
import os
from views import * 
from operator_loader import * 
from pyqtgraph.dockarea import *
import types

import __version__

class ImageSelector(QtGui.QWidget):

  #__pyqtSignals__ = ("latitudeChanged(double)",
 #                    "longitudeChanged(double)")

    def __init__(self, parent = None):

        QtGui.QWidget.__init__(self, parent)

        # current image
        self.imageDesc = 'None'
        self.currentIndex = 0
        self.numInput   = None
        self.batchInputNames = None
        self.ivigraph=None

        self.comboBoxInputSelector = QtGui.QComboBox()
        #imgItem = [str(i) for i in xrange(100)]
        #self.comboBoxInputSelector.addItems(imgItem)

        # prev & next image
        self.prevImgButton = QtGui.QPushButton('<')
        self.nextImgButton = QtGui.QPushButton('>')
        self.prevImgButton.setFixedSize(20,30)
        self.nextImgButton.setFixedSize(20,30)



        layout = QtGui.QVBoxLayout(self)
        layoutB = QtGui.QHBoxLayout()



        layout.addLayout(layoutB)

        layoutB.addWidget(self.comboBoxInputSelector)
        layoutB.addWidget(self.prevImgButton)
        layoutB.addWidget(self.nextImgButton)


        def comboBoxInputSelectorChanged(index):
            self.setCurrentIndex(index)

        self.comboBoxInputSelector.currentIndexChanged.connect(comboBoxInputSelectorChanged)


        def buttonNextImg():
            if self.currentIndex+1 < self.numInput:
                self.setCurrentIndex(self.currentIndex+1)
        self.nextImgButton.released.connect(buttonNextImg)

        def buttonPrevImg():
            if self.currentIndex>0 :
                self.setCurrentIndex(self.currentIndex-1)
        self.prevImgButton.released.connect(buttonPrevImg)


    def setCurrentIndex(self,newIndex):

        self.nextImgButton.setEnabled(newIndex+1<self.numInput)
        self.prevImgButton.setEnabled(newIndex>0)
        self.currentIndex=newIndex
        self.comboBoxInputSelector.setCurrentIndex(newIndex)
        self.ivigraph._updateInput(self.currentIndex)

    def setBatchInputNames(self,ivigraph,batchInputNames):
        names = []
        for i,bn in enumerate(batchInputNames):
            names.append("Img %s:    %s"%(str(i).zfill(3),bn) )

        self.ivigraph = ivigraph
        self.numInput   = len(batchInputNames)
        self.batchInputNames = batchInputNames
        self.comboBoxInputSelector.clear()
        self.comboBoxInputSelector.addItems(names)



class IViGrahp(QtGui.QWidget):
    def __init__(self,parent=None,dataIn=None):

        self.batchInput=None
        self.batchMode=None

        QtGui.QWidget.__init__(self, parent)

        self.win = QtGui.QMainWindow()
        self.win.setWindowTitle("IViGrahp V%s"%__version__.version)
        self.win.setCentralWidget(self)
        self.win.resize(1000,500)

        # ui widgets
        self.mylayout = QtGui.QVBoxLayout(self)
        self.dockArea = DockArea(self)
        self.layout().addWidget(self.dockArea)

        #self.addLayout(self.layout)

 

        self.constrollDock  	= Dock("Controll", size=(1, 1))
        self.flowChartDock    	= Dock("FlowchartDock", size=(1, 1))
        self.viewDocks 			= [	Dock("view0", size=(1, 1)), Dock("view1", size=(1, 1)),Dock("view2", size=(1, 1)), Dock("view3", size=(1, 1)) ]
        self.imageSelectorDock 	= Dock("ImageSelector", size=(1, 1))

        # add docks
        self.dockArea.addDock(self.constrollDock, 'left')
        self.dockArea.addDock(self.viewDocks[0], 'right')
        self.dockArea.addDock(self.viewDocks[3], 'below', self.viewDocks[0])
        self.dockArea.addDock(self.viewDocks[2], 'below', self.viewDocks[0])
        self.dockArea.addDock(self.viewDocks[1], 'below', self.viewDocks[0])
        self.dockArea.addDock(self.flowChartDock, 'below', self.viewDocks[0])
        self.dockArea.addDock(self.imageSelectorDock,'bottom',self.constrollDock)



        # Widgets:
        # viewer widgets
        self.viewers = [ ClickImageView(),ClickImageView(),
            ClickImageView(),ClickImageView()]

        #  flochart (widget)
        self.flowChart = Flowchart(terminals={
        'dataIn': {'io': 'in'},
        'dataOut': {'io': 'out'}    
        })

        # image selctor widget
        self.imgSelector =  ImageSelector(parent=self)
        self.imageSelectorDock.addWidget(self.imgSelector)

        self.flowCharWidget = self.flowChart.widget()
        self.flowCharWidget.ui.showChartBtn.hide()
        self.flowCharWidget.ui.reloadBtn.hide()

        self.constrollDock.addWidget(self.flowCharWidget)
        self.flowChartDock.addWidget(self.flowCharWidget.chartWidget)
        # add widgets
        for viewDock,viewer in zip(self.viewDocks,self.viewers):
            viewDock.addWidget(viewer)

        # add view nodes
        self.viewerNodes = [ self.flowChart.createNode('ImageView', pos=(posX, -150))    for posX in range(0,600,150) ]

        for viewerNode,viewer in zip(self.viewerNodes,self.viewers):
            viewerNode.setView(viewer)


        # fix save and load 
        self.__fixSaveAndLoad()

    def __fixSaveAndLoad(self):

        ivigraph = self

        def setCurrentFileFixed(fcc, fileName):

            fn = str(fileName)
            if fn.endswith('.fc')==False:
                fn+='.fc'

            fcc.currentFileName = fn
            if fn is None:
                fcc.ui.fileNameLabel.setText("<b>[ new ]</b>")
            else:
                fcc.ui.fileNameLabel.setText("<b>%s</b>" % os.path.split(str(fcc.currentFileName))[1])
            fcc.resizeEvent(None)



        def saveFile(self, fileName=None, startDir=None, suggestedFileName='flowchart.fc'):
            
            if fileName is not None:
                fileName = str(fileName)
                if fileName.endswith('.fc')==False:
                    fileName+='.fc'

            if fileName is None:
                if startDir is None:
                    startDir = self.filePath
                if startDir is None:
                    startDir = '.'
                self.fileDialog = pg.FileDialog(None, "Save Flowchart..", startDir, "Flowchart (*.fc)")
                #self.fileDialog.setFileMode(QtGui.QFileDialog.AnyFile)
                self.fileDialog.setAcceptMode(QtGui.QFileDialog.AcceptSave) 
                #self.fileDialog.setDirectory(startDir)
                self.fileDialog.show()
                self.fileDialog.fileSelected.connect(self.saveFile)
                return
                #fileName = QtGui.QFileDialog.getSaveFileName(None, "Save Flowchart..", startDir, "Flowchart (*.fc)")
            configfile.writeConfigFile(self.saveState(), fileName)
            self.sigFileSaved.emit(fileName)

        FlowchartCtrlWidget.setCurrentFile=setCurrentFileFixed
        ivigraph.flowChart.saveFile=types.MethodType(saveFile,ivigraph.flowChart)




        def loadFile(fc, fileName=None, startDir=None):#,nodes=(v1Node,v2Node,v3Node,v4Node),viewers=viewers):
            #print "my file load",type(fc)
            if isinstance(fc,FlowchartCtrlWidget):
                return
            import pyqtgraph.configfile as configfile
            if fileName is None:
                if startDir is None:
                    startDir = fc.filePath
                if startDir is None:
                    startDir = '.'
                fc.fileDialog = pg.FileDialog(None, "Load Flowchart..", startDir, "Flowchart (*.fc)")
                #self.fileDialog.setFileMode(QtGui.QFileDialog.AnyFile)
                #self.fileDialog.setAcceptMode(QtGui.QFileDialog.AcceptSave) 
                fc.fileDialog.show()
                fc.fileDialog.fileSelected.connect(fc.loadFile)
                return
                ## NOTE: was previously using a real widget for the file dialog's parent, but this caused weird mouse event bugs..
                #fileName = QtGui.QFileDialog.getOpenFileName(None, "Load Flowchart..", startDir, "Flowchart (*.fc)")
            fileName = str(fileName)
            state = configfile.readConfigFile(fileName)
            #print "type..",type(fc)
            fc.restoreState(state, clear=True)
            fc.viewBox.autoRange()
            #fc.emit(QtCore.SIGNAL('fileLoaded'), fileName)
            fc.sigFileLoaded.emit(fileName)

            for name, node in fc._nodes.items():
                #print name
                if isinstance(node,ivigraph.viewerNodes[0].__class__):
                    if name =='ImageView.0':
                        node.setView(ivigraph.viewers[0])
                    if name =='ImageView.1':
                        node.setView(ivigraph.viewers[1])
                    if name =='ImageView.2':
                        node.setView(ivigraph.viewers[2])
                    if name =='ImageView.3':
                        node.setView(ivigraph.viewers[3])
            fc.inputNode.update()

        ivigraph.flowChart.loadFile=types.MethodType(loadFile,ivigraph.flowChart)


    def _updateInput(self,index):
        data = vigra.readImage(self.batchInput[index])
        self.flowChart.setInput(dataIn=data)



    def setInput(self,**kwargs):
        self.flowChart.setInput(**kwargs)
        self.batchMode = False
        self.imageSelectorDock.hide()

    def setBatchInput(self,folder,fFilter,inputName='imageIn'):

        self.batchInput      = [ folder + f for f in os.listdir(folder) if f.endswith(fFilter)]
        self.batchInputNames = [ f for f in os.listdir(folder) if f.endswith(fFilter)]

        #for fn in self.batchInput:
        #    print fn

        self.batchMode = True
        self.imgSelector.setBatchInputNames(self,self.batchInputNames)
        self.imgSelector.setCurrentIndex(0)
        self.imageSelectorDock.show()


    