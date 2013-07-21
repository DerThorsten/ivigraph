from pyqtgraph.flowchart import Flowchart, Node,FlowchartCtrlWidget
import pyqtgraph.flowchart.library as fclib
from pyqtgraph.flowchart.library.common import CtrlNode
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg
import numpy as np
import numpy
import scipy.ndimage
import vigra
import os
from views import * 
from operators import * 
from pyqtgraph.dockarea import *
import types

class ImageSelector(QtGui.QWidget):

  #__pyqtSignals__ = ("latitudeChanged(double)",
 #                    "longitudeChanged(double)")

    def __init__(self, parent = None):

        QtGui.QWidget.__init__(self, parent)

        # current image
        self.image     = None
        self.imageDesc = 'None'


        # current image selector
        self.currentImageTextLabel = QtGui.QLabel("CurrentImg:")
        self.currentImageDescLabel = QtGui.QLabel(str(self.currentImgDesc()))

        self.comboBoxFilter = QtGui.QComboBox()
        imgItem = [str(i) for i in xrange(100)]
        self.comboBoxFilter.addItems(imgItem)

        # prev & next image
        self.prevImgButton = QtGui.QPushButton('<')
        self.nextImgButton = QtGui.QPushButton('>')
        self.prevImgButton.setFixedSize(20,30)
        self.nextImgButton.setFixedSize(20,30)

        #
        self.folderSelectorTextLabel = QtGui.QLabel("Folder :")
        self.folderSelector = QtGui.QLineEdit('\\home')
        #self.folderFilterrTextLabel = QtGui.QLabel("Filter :")
        self.folderSelectorFilter = QtGui.QLineEdit('*.png')
        self.loadFolderButton   =QtGui.QPushButton('Load')


        layout = QtGui.QVBoxLayout(self)
        layoutA = QtGui.QHBoxLayout()
        layoutB = QtGui.QHBoxLayout()


        layout.addLayout(layoutA)
        layout.addLayout(layoutB)

        layoutA.addWidget(self.folderSelectorTextLabel)
        layoutA.addWidget(self.folderSelector)
        #layoutA.addWidget(self.folderFilterrTextLabel)
        layoutA.addWidget(self.folderSelectorFilter)
        layoutA.addWidget(self.loadFolderButton)

        layoutB.addWidget(self.currentImageTextLabel)
        layoutB.addWidget(self.currentImageDescLabel)
        layoutB.addWidget(self.prevImgButton)
        layoutB.addWidget(self.nextImgButton)
        layoutB.addWidget(self.comboBoxFilter)

        # on image changed callback
        self.onImageChangedCallBack = None

    def currentImgDesc(self):
        return self.imageDesc

    def connectImageChange(self,f):
        self.onImageChangedCallBack = f

    def onImageChanged(self):
        if self.onImageChangedCallBack is None:
            print "onImageChangedCallBack is none"
        else :
            print "onImageChangedCallBack is called"
            f(self.image,self.iamgeDesc)







class IViGrahp(QtGui.QWidget):
    def __init__(self,parent=None,dataIn=None):
        QtGui.QWidget.__init__(self, parent)

        self.win = QtGui.QMainWindow()

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
        self.imgSelector =  ImageSelector(parent=self.dockArea)
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
            fcc.currentFileName = fileName
            if fileName is None:
                fcc.ui.fileNameLabel.setText("<b>[ new ]</b>")
            else:
                fcc.ui.fileNameLabel.setText("<b>%s</b>" % os.path.split(str(fcc.currentFileName))[1])
            fcc.resizeEvent(None)

        FlowchartCtrlWidget.setCurrentFile=setCurrentFileFixed

        def loadFile(fc, fileName=None, startDir=None):#,nodes=(v1Node,v2Node,v3Node,v4Node),viewers=viewers):
            print "my file load"
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
            fc.restoreState(state, clear=True)
            fc.viewBox.autoRange()
            #fc.emit(QtCore.SIGNAL('fileLoaded'), fileName)
            fc.sigFileLoaded.emit(fileName)

            for name, node in fc._nodes.items():
                print name
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

    def setInput(self,**kwargs):
        self.flowChart.setInput(**kwargs)

