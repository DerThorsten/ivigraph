import pyqtgraph as pg
import numpy as np

_predefColormaps = pg.graphicsItems.GradientEditorItem.Gradients

del _predefColormaps['spectrum']
del _predefColormaps['cyclic']

_predefColormaps['alphaGrey']={'ticks': [(0.0, (0, 0, 0, 0)), (1.0, (255, 255, 255, 255))], 'mode': 'rgb'}


def getColormap(name):
    global _predefColormaps
    rcm  = _predefColormaps[name]

    pp=[]
    cc=[]
    for p,c in rcm['ticks']:
        pp.append(p)
        cc.append(c)
    return pg.ColorMap(pos=np.array(pp),color=np.array(cc))