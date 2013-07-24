import numpy as np
import vigra
from node_base import MyCtrlNode,MyNode
import pyqtgraph.flowchart.library as fclib
from pyqtgraph.flowchart import Node
from pyqtgraph.Qt import QtGui, QtCore


###################################################
#
#   Region Features
#
###################################################


def _generate_region_features_checked_list(checked_true = ['Mean'],
                                           feature_image = np.empty((0,0,0), dtype=np.float32),
                                           label_image = np.empty((0,0,0), dtype=np.uint32)):
    features = vigra.analysis.supportedRegionFeatures(np.require(feature_image, dtype=np.float32),
                                                      np.require(label_image, dtype=np.uint32))
    uiTemplate = []
    for feature in features:
        check_value = False
        if feature in checked_true:
            check_value = True
        uiTemplate.append((feature, 'check', {'checked': check_value}))
    return uiTemplate


def _get_checked_features(check_list):
    checked_features = []
    for key, value in check_list.iteritems():
        if value.isChecked():
            checked_features.append(key)
    return checked_features
    

class RegionFeaturesNode(MyNode):
    """extract region features"""
    nodeName = "RegionFeatures"

    #uiTemplate = []
    uiTemplate = _generate_region_features_checked_list(checked_true = ['Mean'])


    def __init__(self, name):
        terminals = {
            'FeatureImage': dict(io='in'),
            'LabelImage': dict(io='in'),
            'RegionFeatures': dict(io='out'),
            'UsedFeatures': dict(io='out')
        }



        ui = None
        if ui is None:
            if hasattr(self, 'uiTemplate'):
                ui = self.uiTemplate
            else:
                ui = []
        MyNode.__init__(self, name, terminals=terminals, nodeSize=(150,150))
        self.used_features = ['Mean']
        self.ui, self.stateGroup, self.ctrls = fclib.common.generateUi(ui)


    def update(self, signal=True):
        vals = self.inputValues()
        FeatureImage = vals['FeatureImage']
        LabelImage = vals['LabelImage']
        if not (FeatureImage is None or LabelImage is None):
            self.uiTemplate = _generate_region_features_checked_list(checked_true = self.used_features,
                                                                     feature_image = FeatureImage,
                                                                     label_image = LabelImage)
            
            # self.stateGroup.sigChanged.connect(self.changed)

            #gi = self.graphicsItem()


            #self.ui.setParent(gi)
            #action_button = QtGui.QPushButton(text = "Get Features", parent = self.ui)
            #self.ui.layout().addRow(action_button)
            #self.ui.show()
            #action_button.clicked.connect(super(MyCtrlNode, self).update)


    def restoreState(self, state):
        Node.restoreState(self, state)
        if self.stateGroup is not None:
            self.stateGroup.setState(state.get('ctrl', {}))


    def ctrlWidget(self):
        return self.ui

    def hideRow(self, name):
        w = self.ctrls[name]
        l = self.ui.layout().labelForField(w)
        w.hide()
        l.hide()
        
    def showRow(self, name):
        w = self.ctrls[name]
        l = self.ui.layout().labelForField(w)
        w.show()
        l.show()

    def execute(self, FeatureImage, LabelImage, display=True):
        if FeatureImage is None or LabelImage is None:
            return None
        FeatureImage = np.require(FeatureImage, dtype=np.float32)
        LabelImage = np.require(LabelImage, dtype=np.uint32)
        
        self.used_features = _get_checked_features(self.ctrls)
        region_features = vigra.analysis.extractRegionFeatures(FeatureImage,
                                                               LabelImage,
                                                               self.used_features)
        return {
            'RegionFeatures': region_features,
            'UsedFeatures': self.used_features
        }

fclib.registerNodeType(RegionFeaturesNode, [('Image-Analysis',)])
