import numpy as np
import vigra
import math
from node_base import MyCtrlNode,numpyInNumpyOutNode
import pyqtgraph.flowchart.library as fclib





#######################################################
# 
#   TENSOR
#
########################################################

def _tensorEigenvalues(tensor,sortEigenValues,eigenvalue=0):

    ew = vigra.filters.tensorEigenvalues(tensor)
    if  sortEigenValues :
        ew = np.sort(ew,axis=2)
    if eigenvalue<=1:
        return ew[:,:,eigenvalue]
    else :
        return  ew

node = numpyInNumpyOutNode(
    nodeName="BoundaryTensor",
    uiTemplate=[('scale',          'spin', {'value': 1.50, 'step': 0.25, 'range': [0.01, None]})],
    f=vigra.filters.boundaryTensor2D,
    doChannelWise=True,
    tensor=True
)
fclib.registerNodeType(node,[('Image-Tensors',)])


node = numpyInNumpyOutNode(
    nodeName="StructureTensor",
    uiTemplate=[
        ('innerScale',          'spin', {'value': 1.50, 'step': 0.25, 'range': [0.01, None]}),
        ('outerScale',          'spin', {'value': 2.50, 'step': 0.25, 'range': [0.01, None]}),
    ],
    f=vigra.filters.structureTensor
)
fclib.registerNodeType(node,[('Image-Tensors',)])

node = numpyInNumpyOutNode(
    nodeName="LaplacianOfGaussian",
    uiTemplate=[
        ('scale','spin', {'value': 1.50, 'step': 0.25, 'range': [0.01, None]})
    ],
    f=vigra.filters.laplacianOfGaussian
)
fclib.registerNodeType(node,[('Image-Tensors',)])


node = numpyInNumpyOutNode(
    nodeName="TensorEigenvalues",
    uiTemplate=[
        ('sortEigenValues', 'check', {'value': True}),
        ('eigenvalue', 'combo', {'values': ['first', 'second','both'], 'index': 0})
    ],
    f=_tensorEigenvalues
)

fclib.registerNodeType(node,[('Image-Tensors',)])


class StructureTensorTrace(MyCtrlNode):
    """ structure tensor trace"""
    nodeName = "StructureTensorTrace"

    uiTemplate=[('innerscale', 'spin', {'value': 1.5, 'step': 0.1, 'range': [0.01, None]}),
                ('outerscale', 'spin', {'value': 2.5, 'step': 0.1, 'range': [0.01, None]})]
    def __init__(self, name):
        terminals = {
            'Image': dict(io='in'),     
            'dataOut': dict(io='out')  # to specify whether it is input or output
        }                              # other more advanced options are available
                                       # as well..
        MyCtrlNode.__init__(self, name, terminals=terminals)
    def execute(self, Image, display=True):
        
        tensorValues = vigra.filters.structureTensor(image=Image,
                                                     innerScale = self.ctrls['innerscale'].value(),
                                                     outerScale = self.ctrls['outerscale'].value())
        out = np.zeros(tensorValues.shape[:2])
        stepSize = math.sqrt(2*tensorValues.shape[2] + 0.25) - 0.5
        assert stepSize == math.floor(stepSize)
        index = 0
        while (stepSize > 0):
            out += tensorValues[..., index]
            index += stepSize
            stepSize -= 1
        
        return {'dataOut': out}
        

        
fclib.registerNodeType(StructureTensorTrace ,[('Image-Tensors',)])



class TensorTrace(MyCtrlNode):
    """ calculate trace of tensor input """
    nodeName = "TensorTrace"

    def __init__(self, name):
        terminals = {
            'Tensor': dict(io='in'),
            'dataOut': dict(io='out')
        }

        MyCtrlNode.__init__(self, name, terminals=terminals)
    def execute(self, Tensor, display=True):
        
        assert len(Tensor.shape) == 3
        out = np.zeros(Tensor.shape[:2])
        stepSize = math.sqrt(2*Tensor.shape[2] + 0.25) - 0.5
        assert stepSize == math.floor(stepSize)
        index = 0
        while (stepSize > 0):
            out += Tensor[..., index]
            index += stepSize
            stepSize -= 1
        
        return {'dataOut': out}


fclib.registerNodeType(TensorTrace, [('Image-Tensors',)])

