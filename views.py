from pyqtgraph.flowchart import Flowchart, Node
import pyqtgraph.flowchart.library as fclib
from pyqtgraph.flowchart.library.common import CtrlNode
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg
import numpy
import scipy.ndimage
import vigra



class ClickImageView(pg.ImageView):
    def __init__(self,*args,**kwargs):
        super(ClickImageView, self).__init__(*args,**kwargs)
        self.clicks=[]
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

    def keyPressEvent(self, ev):
        print ev.key()
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
        ('clearClicks', 'check', {'value': True,'isChecked':True})
    ]
    def __init__(self, name):
        self.view = None
        ## Initialize node with only a single input terminal
        CtrlNode.__init__(  self, name, 
                            terminals={
                                'data'  : {'io':'in'},
                                'data2' : {'io':'in'},
                                'view': {'io':'out'}
                            }
        )

    def setView(self, view):  ## setView must be called by the program
        self.view = view
        

    def processEvents(self):
        clearClicks = self.ctrls['clearClicks'].isChecked()

        if(clearClicks):
            self.view.clicks = []
        else:
            print "num clicks",len(self.view.clicks)


    def process(self, data,data2=None, display=True):
        self.processEvents()

        alpha = self.ctrls['alpha'].value()
        displayType = ImageViewNode.imageTypes[self.ctrls['imageType'].currentIndex()]

        if (data2 is not None) and alpha > 0.0:
            data2=numpy.squeeze(data2)
            d2=data2.copy()
            normBlend = self.ctrls['normalizeBlendChannel'].isChecked()
            print "normbBlend",normBlend
            if (normBlend):
                d2-=d2.min()
                d2/=d2.max()
                d2*=255.0

            dim1=data.ndim
            dim2=d2.ndim

            if dim1==3 and dim2==3 :
                print "do colorblending"
                dataBlended = (1.0-alpha)*data + alpha*(d2)
                
            if dim1==3 and dim2==2:
                print "do color gray bleding"
                dataBlended =data.copy()
                for c in range(3):
                    dataBlended[:,:,c] = (1.0-alpha)*data[:,:,c] + alpha*d2

            if dim1==2 and dim2==3:
                assert False

            self.view.setImage(dataBlended)
        else:        


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
                        self.view.setImage(data)
                    elif(ndim==3):
                        print "display some multichannel"
                        # number of channels
                        nC = data.shape[2]
                        if displayType=='RgbType':
                            print "display RGB"
                            # check channels
                            if nC !=3 : 
                                raise RuntimeError("wrong number of channels for rgb image")
                            self.view.setImage(data)
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
                            self.view.setImage(data[:,:,channel])
                        else:
                            assert False

        return {'view': self.view}


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