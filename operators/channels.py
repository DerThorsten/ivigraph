import numpy as np
import vigra
import math
from collections import OrderedDict

from node_base import MyNode,MyCtrlNode,numpyInNumpyOutNode
import pyqtgraph.flowchart.library as fclib






##################################################
#
#   CHANNELS
#
###################################################
class ChannelStacker(MyNode):
    """ stack channels of all input(s) """
    nodeName = "ChannelStacker"

    uiTemplate=[
    ]

    def __init__(self, name):
        """
        terminals = {
            'Data0': dict(io='in'),
            'Data1': dict(io='in'),
            'dataOut': dict(io='out'),
        }p
        """

        terminals = OrderedDict()
        terminals['Input']=dict(io='in')
        terminals['dataOut']=dict(io='out')
        MyNode.__init__(self, name, terminals=terminals,nodeSize=(100,150),allowAddInput=True)


    def execute(self, *args,**kwargs):

        inputList = []
        inputShape = None
        # iterate over all terminals in ordere they where added
        for termName in self.terminals.keys():
            term = self.terminals[termName]
            if termName in self._inputs:
                inputData  =  kwargs[termName]
                if inputData is not None:
                    data  = np.array(inputData,dtype=np.float32)
                    iShape = data.shape
                    reshapedInput = data.reshape([iShape[0],iShape[1],-1])
                    inputList.append(reshapedInput)
        
        if len(inputList)>0:
            stacked =  np.concatenate(inputList,axis=2)
            print "outShape",stacked.shape
        else:
            stacked = None
        return {'dataOut':stacked}


fclib.registerNodeType(ChannelStacker, [('Image-Channels',)])

def _channelAcc(image,accumulation):

    if(accumulation==0):
        return np.sum(image,axis=2)
    elif(accumulation==1):
        return np.product(image,axis=2)
    elif(accumulation==2):
        return np.min(image,axis=2)
    elif(accumulation==3):
        return np.max(image,axis=2)
    elif(accumulation==4):
        return np.mean(image,axis=2)
    elif(accumulation==5):
        return np.median(image,axis=2)
    else:
        assert False

def _channelSelector(image,channel):
    return image[:,:,channel]


node = numpyInNumpyOutNode(
    nodeName="ChannelAccumuation",
    uiTemplate=[
        ('accumulation', 'combo', {'values': ['sum', 'product','min','max','mean','median'], 'index': 0})
    ],
    f=_channelAcc
)

fclib.registerNodeType(node,[('Image-Channels',)])


node = numpyInNumpyOutNode(
    nodeName="ChannelSelector",
    uiTemplate=[
       ('channel','intSpin', {'value': 0, 'min': 0, 'max': 1e9 })
    ],
    f=_channelSelector
)

fclib.registerNodeType(node,[('Image-Channels',)])



###################################################
#
#   Blender
#
###################################################
class Blender(MyCtrlNode):
    """ blend images (weighted), normalize, if neccessary """
    nodeName = "Blender"

    uiTemplate=[('normalize', 'combo', {'values': ['0', '1', '255']}),
                ('weight', 'spin', {'value' : 0.5, 'step' : 0.05, 'range': [0.0, 1.0]})]

    def __init__(self, name):
        terminals = {
            'Image1': dict(io='in'),
            'Image2': dict(io='in'),
            'BlendedImage': dict(io='out')
        }
        MyCtrlNode.__init__(self, name, terminals=terminals)
    def execute(self, Image1, Image2, display=True):
        if Image1 is None or Image2 is None:
            return {'BlendedImage': np.zeros((0,0))}
        if len(Image1.shape) == 2:
            Image1 = vigra.VigraArray(Image1[..., np.newaxis],
                                      axistags = vigra.VigraArray.defaultAxistags(3))
        if len(Image2.shape) == 2:
            Image2 = vigra.VigraArray(Image2[..., np.newaxis],
                                      axistags = vigra.VigraArray.defaultAxistags(3))
        if Image1.shape[:2] != Image2.shape[:2]:
            raise Exception("Image dimensions disagree!")
        if Image1.shape[2] == 1 and Image2.shape[2] == 3:
            Image1 = np.repeat(Image1, 3, axis=2)
        elif Image1.shape[2] == 3 and Image2.shape[2] == 1:
            Image2 = np.repeat(Image2, 3, axis=2)
        if Image1.shape[2] != Image2.shape[2]:
            raise Exception("Image channels disagree!")

        normalization = int(self.ctrls['normalize'].currentText())
        if normalization > 0:
            Image1 = _normalize(Image1, 0, normalization)
            Image2 = _normalize(Image2, 0, normalization)
        weight = self.ctrls['weight'].value()
        return {'BlendedImage': weight*Image1 + (1-weight)*Image2}
fclib.registerNodeType(Blender, [('Image-Channels',)])