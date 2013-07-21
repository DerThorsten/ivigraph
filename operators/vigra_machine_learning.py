import numpy as np
import vigra
import math
from collections import OrderedDict
from scipy.stats.mstats import mquantiles

from node_base import convertNh, MyNode,MyCtrlNode,numpyInNumpyOutNode
import pyqtgraph.flowchart.library as fclib




###################################################
#
#   Random Forest
#
###################################################


class RandomForest(MyCtrlNode):
    """ blend images (weighted), normalize, if neccessary """
    nodeName = "RandomForest"

    uiTemplate=[
        ('predictLabels', 'check',   {'value': True}),
        ('predictProbs', 'check',   {'value': True}),
        ('numTrees','intSpin', {'value': 10, 'min': 0, 'max': 1e9 })
    ]

    def __init__(self, name):
        terminals = {
            'Features': dict(io='in'),
            'Labels': dict(io='in'),
            'RF': dict(io='out'),
            'PredictedLabels': dict(io='out'),
            'PredictedProbs': dict(io='out'),
            'OOB-Error': dict(io='out'),
        }
        MyCtrlNode.__init__(self, name, terminals=terminals,nodeSize=(150,150))
    def process(self, Features, Labels, display=True):
        self.startProcess()
        numFeatures = Features.shape[2]

        shape = Labels.shape
        features = Features.astype(np.float32).reshape([-1,numFeatures])
        labels   = Labels.astype(np.uint32).reshape(-1)


        where1 = np.where(labels==1)
        where2 = np.where(labels==2)

        print where1
        print where2

        f1  = features[where1[0],:]
        f2  = features[where2[0],:]
        f= np.concatenate([f1,f2],axis=0).astype(np.float32)

        print "f1 shape",f1.shape
        print "f2 shape",f2.shape
        print "f shape",f.shape

        numLabeledPoints = f.shape[0]

        print "numLabeledPoints",numLabeledPoints

        l= np.zeros([f.shape[0],1],dtype=np.uint32)
        l[0:len(where1[0])]=0
        l[len(where1[0]):len(where1[0])+len(where2[0])]=1

        print "learn rf"
        print f.shape , l .shape 
        rf=vigra.learning.RandomForest(treeCount=self.ctrls['numTrees'].value())
        oob = rf.learnRF(f,l)
        print "learn rf done",oob

        print "predict"
        probs = rf.predictProbabilities(features)
        print "predict done"
        probsImg = np.array(probs)
        probsImg = probsImg.reshape(tuple(shape)+(2,))

        self.endProcess()
        return {
            'RF': None,
            'PredictedLabels' : None,
            'PredictedProbs'  : probsImg,
            'OOB-Error'       : None
        }

fclib.registerNodeType(RandomForest, [('Image-MachineLearning',)])




###################################################
#
#   Learn Random Forest
#
###################################################

class LearnRandomForest(MyCtrlNode):
    """ train a random forest classifier """
    nodeName = "LearnRandomForest"

    uiTemplate=[
        ('number of trees', 'intSpin', {'value': 100, 'min': 0, 'max': 1e9 })
    ]

    def __init__(self, name):
        terminals = {
            'FeatureMatrix': dict(io='in'),
            'LabelVector': dict(io='in'),
            'RandomForest': dict(io='out'),
            'OOB-Error': dict(io='out')
        }
        MyCtrlNode.__init__(self, name, terminals=terminals)
    def process(self, FeatureMatrix, LabelVector, display=True):
        self.startProcess()
        if FeatureMatrix is None or LabelVector is None:
            return
        if FeatureMatrix.shape[0] != LabelVector.shape[0]:
            raise Exception("number of labels does not agree with feature matrix")
        if FeatureMatrix.dtype is not np.float32:
            FeatureMatrix = FeatureMatrix.astype(np.float32)
        if LabelVector.dtype is not np.uint32:
            LabelVector = LabelVector.astype(np.uint32)
        rf = vigra.learning.RandomForest(treeCount=self.ctrls['number of trees'].value())
        oob = rf.learnRF(FeatureMatrix, LabelVector)
        self.endProcess()
        return {'RandomForest': rf, 'OOB-Error': oob}

fclib.registerNodeType(LearnRandomForest, [('Image-MachineLearning',)])

###################################################
#
#   Predict Random Forest
#
###################################################

class PredictRandomForest(MyCtrlNode):
    """ predict labels using a random forest classifier """
    nodeName = "PredictRandomForest"

    def __init__(self, name):
        terminals = {
            'FeatureMatrix': dict(io='in'),
            'RandomForest': dict(io='in'),
            'Predictions': dict(io='out')
        }
        MyCtrlNode.__init__(self, name, terminals=terminals)
    def process(self, FeatureMatrix, RandomForest, display=True):
        self.startProcess()
        if FeatureMatrix is None:
            return
        if FeatureMatrix.dtype is not np.float32:
            FeatureMatrix = FeatureMatrix.astype(np.float32)
        predictions = RandomForest.predictProbabilities(FeatureMatrix)
        self.endProcess()
        return {'Predictions': predictions}

fclib.registerNodeType(PredictRandomForest, [('Image-MachineLearning',)])


###################################################
#
#   PredictionToImage
#
###################################################

class PredictionToImage(MyCtrlNode):
    """ reshape predictions to image """
    nodeName = "PredictionToImage"

    uiTemplate=[
        ('probability for class', 'intSpin', {'value': 1, 'min': 1, 'max': 1e9 })
    ]
    def __init__(self, name):
        terminals = {
            'Predictions': dict(io='in'),
            'ImageForShape': dict(io='in'),
            'Image': dict(io='out')
        }
        MyCtrlNode.__init__(self, name, terminals=terminals)
    def process(self, Predictions, ImageForShape, display=True):
        self.startProcess()
        if Predictions is None or ImageForShape is None:
            return
        print ImageForShape.shape
        print Predictions.shape
        classLabel = self.ctrls['probability for class'].value()
        Image = Predictions[...,classLabel-1].reshape(ImageForShape.shape, order='F')
        self.endProcess()
        return {'Image': Image}

fclib.registerNodeType(PredictionToImage, [('Image-MachineLearning',)])







###################################################
#
#   FeatureStackToMatrix
#
###################################################
class FeatureStackToMatrix(MyCtrlNode):
    """ create feature matrix from labels and feature image """
    nodeName = "FeatureStackToMatrix"

    uiTemplate = [('no label', 'spin', {'value': 0, 'step' : 1, 'range' : [0, None]})]

    def __init__(self, name):
        terminals = {
            'LabelImage': dict(io='in'),
            'Features': dict(io='in'),
            'FeatureMatrix': dict(io='out'),
            'LabelVector': dict(io='out')
        }
        MyCtrlNode.__init__(self, name, terminals=terminals)
    def process(self, LabelImage, Features, display=True):
        self.startProcess()
        if Features is None or \
           np.all(LabelImage == 0):
            return {'FeatureMatrix': None, 'LabelVector': None}
        no_label_value = self.ctrls['no label'].value()

        # only for 2D -> workaraound for 3D?
        if len(Features.shape) == 2:
            Features = Features[..., np.newaxis]
        
        if LabelImage is not None:
            label_indices = np.where(LabelImage[...] != no_label_value)
            label_vector = LabelImage[label_indices]
            label_vector = label_vector[..., np.newaxis]
            number_of_samples = label_indices[0].shape[0]
        else:
            label_indices = (slice(0, Features.shape[0]),
                             slice(0, Features.shape[1]))
            label_vector = None
            number_of_samples = Features.shape[0]*Features.shape[1]
        number_of_features = Features.shape[2]
        feature_matrix = np.empty((number_of_samples, number_of_features))
        for index in xrange(Features.shape[-1]):
            feature = Features[...,index]
            feature_matrix[..., index] = feature[label_indices].flatten()
        self.endProcess()
        return {'FeatureMatrix': feature_matrix, 'LabelVector': label_vector}
            
    
    def graphicsItem(self):
        if self._graphicsItem is None:
            self._graphicsItem = CustomNodeGraphicsItem(self,(200,100))
        return self._graphicsItem


fclib.registerNodeType(FeatureStackToMatrix, [('Image-MachineLearning',)])