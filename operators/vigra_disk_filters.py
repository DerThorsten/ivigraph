import numpy as np
import vigra
import math
from node_base import numpyInNumpyOutNode
import pyqtgraph.flowchart.library as fclib
#######################################################
# 
#   DISK-Filters
#
########################################################
node = numpyInNumpyOutNode(
    nodeName="DiscClosing",
    uiTemplate=[
        ('radius','intSpin', {'value': 1, 'min': 1, 'max': 1e9 })
    ],
    f=vigra.filters.discClosing,
    dtypeIn=np.uint8
)
fclib.registerNodeType(node,[('Image-Disc-Filters',)])


node = numpyInNumpyOutNode(
    nodeName="DiscDilation",
    uiTemplate=[
        ('radius','intSpin', {'value': 1, 'min': 1, 'max': 1e9 })
    ],
    f=vigra.filters.discDilation,
    dtypeIn=np.uint8
)
fclib.registerNodeType(node,[('Image-Disc-Filters',)])
    
node = numpyInNumpyOutNode(
    nodeName="DiscErosion",
    uiTemplate=[
        ('radius','intSpin', {'value': 1, 'min': 1, 'max': 1e9 })
    ],
    f=vigra.filters.discErosion,
    dtypeIn=np.uint8
)
fclib.registerNodeType(node,[('Image-Disc-Filters',)])


node = numpyInNumpyOutNode(
    nodeName="DiscMedian",
    uiTemplate=[
        ('radius','intSpin', {'value': 1, 'min': 1, 'max': 1e9 })
    ],
    f=vigra.filters.discMedian,
    dtypeIn=np.uint8
)
fclib.registerNodeType(node,[('Image-Disc-Filters',)])



node = numpyInNumpyOutNode(
    nodeName="DiscOpening",
    uiTemplate=[
        ('radius','intSpin', {'value': 1, 'min': 1, 'max': 1e9 })
    ],
    f=vigra.filters.discOpening,
    dtypeIn=np.uint8
)
fclib.registerNodeType(node,[('Image-Disc-Filters',)])


node = numpyInNumpyOutNode(
    nodeName="DiscRankOrderFilter",
    uiTemplate=[
        ('radius','intSpin', {'value': 1, 'min': 1, 'max': 1e9 }),
        ('rank','spin', {'value': 0.50, 'step': 0.1, 'range': [0.0, 1.0]})
    ],
    f=vigra.filters.discRankOrderFilter,
    dtypeIn=np.uint8
)
fclib.registerNodeType(node,[('Image-Disc-Filters',)])