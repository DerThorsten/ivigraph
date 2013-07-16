from pyqtgraph.flowchart import Flowchart, Node
import pyqtgraph.flowchart.library as fclib
from pyqtgraph.flowchart.library.common import CtrlNode
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg
import numpy
import scipy.ndimage
import vigra


import numpy as np
import collections




class ImageAndLabelItem(pg.ImageItem):
    def __init__(self,*args,**kwargs):
        super(ImageAndLabelItem, self).__init__(*args,**kwargs)
        self.clicks=[]

        self.labelImage = None 
        print "non setter has been called"

    def redrawLabels(self):
        print "in redraw labels"

        for l in range(1,self.numLabels):
            print "l",l

            print "self.labelImage",self.labelImage.shape
            print "self.imag",self.image.shape
            print "0"
            whereL = numpy.where(self.labelImage==l)
            nl =len(whereL[0])
            print "nl",nl
            if nl>0:
                print "1"
                self.clickImageView.setCurrentLabel(l)
                print "2"
                self.image[whereL[0],whereL[1],:]=self.clickImageView.labelColors[l]
                print "3"

    def drawAt(self, pos, ev=None):
        print " drawing label",self.currentLabel
        if self.labelImage is None:
            print "set up label image"
            self.labelImage = numpy.zeros( [ self.image.shape[0],self.image.shape[1]])


        pos = [int(pos.x()), int(pos.y())]
        dk = self.drawKernel
        kc = self.drawKernelCenter
        sx = [0,dk.shape[0]]
        sy = [0,dk.shape[1]]
        tx = [pos[0] - kc[0], pos[0] - kc[0]+ dk.shape[0]]
        ty = [pos[1] - kc[1], pos[1] - kc[1]+ dk.shape[1]]
        
        for i in [0,1]:
            dx1 = -min(0, tx[i])
            dx2 = min(0, self.image.shape[0]-tx[i])
            tx[i] += dx1+dx2
            sx[i] += dx1+dx2

            dy1 = -min(0, ty[i])
            dy2 = min(0, self.image.shape[1]-ty[i])
            ty[i] += dy1+dy2
            sy[i] += dy1+dy2

        ts = (slice(tx[0],tx[1]), slice(ty[0],ty[1]))
        ss = (slice(sx[0],sx[1]), slice(sy[0],sy[1]))
        mask = self.drawMask
        src = dk
        
        if isinstance(self.drawMode, collections.Callable):
            self.drawMode(dk, self.image, mask, ss, ts, ev)
        else:
            src = src[ss]
            self.labelImage[ts]=self.currentLabel
            if self.drawMode == 'set':
                if mask is not None:
                    mask = mask[ss]
                    self.image[ts] = self.image[ts] * (1-mask) + src * mask
                else:
                    self.image[ts] = src
            elif self.drawMode == 'add':
                self.image[ts] += src
            elif self.drawMode == 'label':
                self.image[ts] = self.currentLabelColor
            else:
                raise Exception("Unknown draw mode '%s'" % self.drawMode)
            self.updateImage()

        #self._viewNode.update()

        #for d in self._viewNode.dependentNodes():
        #   d.update()


    def setDrawKernel(self, kernel=None, mask=None, center=(0,0), mode='set'):
        self.drawKernel = kernel
        self.drawKernelCenter = center
        self.drawMode = mode
        self.drawMask = mask





class ClickImageView(pg.ImageView):
    def __init__(self,*args,**kwargs):
        super(ClickImageView, self).__init__(imageItem=ImageAndLabelItem())

        self.clicks=[]
        self.currentLabel=1
        self.imageItem.currentLabel=1
        self.numLabels=10
        self.imageItem.numLabels=10
       
        self.imageItem.clickImageView=self

        self.brushSize = 3

        self.labelColors =numpy.ones([self.numLabels,3])
        self.labelColors[0,:]=0,0,0
        self.labelColors[1,:]=1,0,0
        self.labelColors[2,:]=0,1,0
        self.labelColors[3,:]=0,0,1

        self.labelColors[4,:]=0.8,0.5,0.0
        self.labelColors[5,:]=0.5,0.8,0.0
        self.labelColors[6,:]=0.0,0.5,0.8


        self.labelColors[7,:]=0.8,0.4,0.2
        self.labelColors[8,:]=0.4,0.8,0.3
        self.labelColors[9,:]=0.2,0.4,0.8



        self.labelColors*=255.0

        self.bufferImage = None

    def toBuffer(self,image):
        if self.bufferImage is None :
            print "first copy"
            self.bufferImage=image.copy()
        elif image.ndim == self.bufferImage.ndim and tuple(image.shape) == tuple(self.bufferImage.shape):
            print "no alloc  copy"
            self.bufferImage[:,:,:]=image[:,:,:]
        else:
            print "not matching => alloc  copy"
            self.bufferImage=image.copy()


    def keyReleaseEvent(self, ev):
        if ev.key() in [QtCore.Qt.Key_Space, QtCore.Qt.Key_Home, QtCore.Qt.Key_End]:
            ev.accept()
        elif ev.key() in self.noRepeatKeys:
            ev.accept()
            if ev.isAutoRepeat():
                return
            try:
                del self.keysPressed[ev.key()]
            except:
                self.keysPressed = {}
            self.evalKeyState()
        else:
            QtGui.QWidget.keyReleaseEvent(self, ev)

    def setCurrentLabel(self,label):
        self.currentLabel = label
        self.imageItem.currentLabel = label 
        self.imageItem.currentLabelColor = self.labelColors[self.currentLabel,:]
        for x in xrange(self.imageItem.drawKernel.shape[0]):
            for y in xrange(self.imageItem.drawKernel.shape[1]):
                self.imageItem.drawKernel[x,y,:]=self.labelColors[self.currentLabel,:]

    def keyPressEvent(self, ev):
        print ev.key()



        if ev.key()  == QtCore.Qt.Key_U:
            print "update dependentNodes"
            for d in self._viewNode.dependentNodes():
               d.update()

        if ev.key()  == QtCore.Qt.Key_0:
            self.setCurrentLabel(0)
        elif ev.key()  == QtCore.Qt.Key_1:
            self.setCurrentLabel(1)
        elif ev.key()  == QtCore.Qt.Key_2:
            self.setCurrentLabel(2)
        elif ev.key()  == QtCore.Qt.Key_3:
            self.setCurrentLabel(3)
        elif ev.key()  == QtCore.Qt.Key_4:
            self.setCurrentLabel(4)
        elif ev.key()  == QtCore.Qt.Key_5:
            self.setCurrentLabel(5)
        elif ev.key()  == QtCore.Qt.Key_6:
            self.setCurrentLabel(6)
        elif ev.key()  == QtCore.Qt.Key_7:
            self.setCurrentLabel(7)
        elif ev.key()  == QtCore.Qt.Key_8:
            self.setCurrentLabel(8)
        elif ev.key()  == QtCore.Qt.Key_9:
            self.setCurrentLabel(9)


        print "CURRENT LABEL ",self.currentLabel

        if ev.key() == QtCore.Qt.Key_Space:
            if self.playRate == 0:
                fps = (self.getProcessedImage().shape[0]-1) / (self.tVals[-1] - self.tVals[0])
                self.play(fps)
                #print fps
            else:
                self.play(0)
            ev.accept()
        elif ev.key() == QtCore.Qt.Key_Home:
            self.setCurrentIndex(0)
            self.play(0)
            ev.accept()
        elif ev.key() == QtCore.Qt.Key_End:
            self.setCurrentIndex(self.getProcessedImage().shape[0]-1)
            self.play(0)
            ev.accept()
        elif ev.key() in self.noRepeatKeys:
            ev.accept()
            if ev.isAutoRepeat():
                return
            self.keysPressed[ev.key()] = 1
            self.evalKeyState()
        else:
            QtGui.QWidget.keyPressEvent(self, ev)


    """
    def mousePressEvent(self, event):
        if self.image is None:
            return

        button = event.button()
        print button
        if button not in [ 1 , 2 ] :
            x = float(event.x())
            y = float(event.y())
            imgBB = self.imageItem.sceneBoundingRect()
            if imgBB.contains(x,y):
                sx = (x-imgBB.x())/float(imgBB.width())*self.image.shape[0]
                sy = (y-imgBB.y())/float(imgBB.height())*self.image.shape[1]
                if self.clicks is None:
                    assert False
                else :
                    self.clicks.append((sx,sy))
            else :
                pass
    """
    def enableLabelMode(self,viewNode=None,size=1,setLevels=True):        
        kern = numpy.ones([size,size,3])
        for x in xrange(kern.shape[0]):
            for y in xrange(kern.shape[1]):
                kern[x,y,:]=self.labelColors[self.currentLabel,:]
        self.imageItem.currentLabelColor = self.labelColors[self.currentLabel,:]
        self.imageItem.setDrawKernel(kern, mask=kern, center=(int(size)/2,int(size)/2), mode='label')
        self.setCurrentLabel(1)
        if setLevels :
            self.imageItem.setLevels([0, 10])
        if viewNode is not None:
            self._viewNode = viewNode





## At this point, we need some custom Node classes since those provided in the library
## are not sufficient. Each node will define a set of input/output terminals, a 
## processing function, and optionally a control widget (to be displayed in the 
## flowchart control panel)

class ImageViewNode(CtrlNode):
    """Node that displays image data in an ImageView widget"""
    nodeName = 'ImageView'

    imageTypes  = ['RgbType','LabType','MultiChannel']
    uiTemplate=[
        ('imageType', 'combo', {'values': imageTypes, 'index': 0}),
        ('channel','intSpin', {'value': 0, 'min': 0, 'max': 1e9 }),
        ('alpha','spin', {'value': 0,'step':0.25,'range': [0.0, 1.0] }),
        ('normalizeBlendChannel', 'check', {'value': True,'isChecked':True}),
        ('clearClicks', 'check', {'value': True,'isChecked':True}),
        ('brushSize','intSpin', {'value': 5, 'min': 1, 'max': 99 })
    ]
    def __init__(self, name):
        self.view = None
        ## Initialize node with only a single input terminal
        CtrlNode.__init__(  self, name, 
                            terminals={
                                'data'  : {'io':'in'},
                                'data2' : {'io':'in'},
                                'view':    {'io':'out'},
                                'labelImage':{'io':'out'}
                            }
        )

    def setView(self, view):  ## setView must be called by the program
        self.view = view
        self.view.enableLabelMode(self)

    def processEvents(self):
        clearClicks = self.ctrls['clearClicks'].isChecked()

        if(clearClicks):
            self.view.clicks = []
        else:
            print "num clicks",len(self.view.clicks)


    def process(self, data,data2=None, display=True):

        if data.ndim==3 and data.shape[2]==3:

            for c in range(3):
                data[:,:,c]-=data[:,:,c].min()
                data[:,:,c]/=data[:,:,c].max()
                data[:,:,c]*=255.0

        self.view.toBuffer(data)

        self.processEvents()
        alpha = self.ctrls['alpha'].value()
        displayType = ImageViewNode.imageTypes[self.ctrls['imageType'].currentIndex()]
        brushSize =self.ctrls['brushSize'].value()

        self.view.enableLabelMode(size=brushSize,setLevels=False)

        self.view.imageItem.brushSize=brushSize




        ## if process is called with display=False, then the flowchart is being operated
        ## in batch processing mode, so we should skip displaying to improve performance.
        if display and self.view is not None:

            ## the 'data' argument is the value given to the 'data' terminal
            if data is None:
                self.view.setImage(np.zeros((1,1))) # give a blank array to clear the view
            else:
                #m get constrolls
                channel     = self.ctrls['channel'].value()
                displayType = ImageViewNode.imageTypes[self.ctrls['imageType'].currentIndex()]

                # dimensions
                ndim =data.ndim 
                if(ndim==2 or (ndim==3 and data.shape[2]==1) ):
                    print "display grayscaled"
                    self.view.setImage(data.copy())
                elif(ndim==3):
                    print "display some multichannel"
                    # number of channels
                    nC = data.shape[2]
                    if displayType=='RgbType':
                        print "display RGB"
                        # check channels
                        if nC !=3 : 
                            raise RuntimeError("wrong number of channels for rgb image")
                        self.view.setImage(data.copy())
                    elif displayType=='LabType':
                        print "display LAB"
                        # check channels
                        if nC !=3 : 
                            raise RuntimeError("wrong number of channels for lab image")
                        self.view.setImage(vigra.colors.transform_Lab2RGB(data))
                    elif displayType=='MultiChannel':
                        print "display MultiChannel"
                        # check SELECTED channels
                        if channel >= nC :
                            raise RuntimeError("channel index is out of bounds")
                        self.view.setImage(data[:,:,channel].copy())
                    else:
                        assert False


        print "redraw labels?"
        if self.view.imageItem.labelImage is not None:
            self.view.imageItem.redrawLabels()

        #self.autoRange()
        #self.autoLevels()
        return {'view': self.view,'labelImage':self.view.imageItem.labelImage}


## register the class so it will appear in the menu of node types.
## It will appear in the 'display' sub-menu.
fclib.registerNodeType(ImageViewNode, [('Display',)])



"""
print "name ",pName ,"pType",pType
if(pType=='spin' or pType == 'intSpin'):
    kwargs[pName]=self.ctrls[pName].value()
elif(pType=='check'):
    kwargs[pName]=self.ctrls[pName].isChecked()
elif(pType=='combo'):
    kwargs[pName]=self.ctrls[pName].currentIndex()
"""                    