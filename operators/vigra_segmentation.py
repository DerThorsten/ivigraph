import numpy as np
import vigra
import math
from collections import OrderedDict

from node_base import convertNh,MyCtrlNode,numpyInNumpyOutNode
import pyqtgraph.flowchart.library as fclib





#######################################################
# 
#   SEGMETNATION
#
########################################################

class SegVisu(MyCtrlNode):
    """ visualization of a segmentaion"""
    nodeName = "SegVisu"

    uiTemplate=[
        #('neighborhood', 'combo', {'values': ['4', '8'], 'index': 0})
    ]

    def __init__(self, name):
        terminals = OrderedDict()
        terminals['labelImage']=dict(io='in')
        terminals['image']=dict(io='in')
        terminals['dataOut']=dict(io='out')
        # as well..
        MyCtrlNode.__init__(self, name, terminals=terminals)


    def execute(self, labelImage,image=None, display=True):
        #nh=4
        #if self.ctrls['neighborhood'].currentIndex() == 1 :
        #    nn=8

        lImg = np.require(labelImage,dtype=np.uint32)

        crackedEdgeImage = vigra.analysis.regionImageToCrackEdgeImage(lImg)

        whereNoEdge = np.where(crackedEdgeImage!=0)
        whereEdge   = np.where(crackedEdgeImage==0)
        crackedEdgeImage[np.where(crackedEdgeImage!=0)]=1


        if image  is not None :
            if image.ndim==3 :
                if tuple(image.shape[0:2]) == tuple(crackedEdgeImage.shape):
                    imgOut=image.copy()
                else:
                    imgOut=vigra.sampling.resize(image,tuple(crackedEdgeImage.shape))

                for c in range(imgOut.shape[2]):
                    imgOut[whereEdge[0],whereEdge[1],c]=0.0

            else :
                if tuple(image.shape[0:2]) == tuple(crackedEdgeImage.shape):
                    imgOut=image.copy()
                else:
                    imgOut=vigra.sampling.resize(image,tuple(crackedEdgeImage.shape))
                imgOut[whereEdge[0],whereEdge[1]]=0.0

            return {'dataOut': imgOut}
        
fclib.registerNodeType(SegVisu ,[('Image-Segmentation',)])







class Watershed(MyCtrlNode):
    """ (seeded) watershed"""
    nodeName = "Watershed"

    uiTemplate=[('neighborhood', 'combo', {'values': ['4', '8'], 'index': 0})]

    def __init__(self, name):
        terminals = {
            'growImage': dict(io='in'),
            'seedImage': dict(io='in'),     
            'dataOut': dict(io='out')  # to specify whether it is input or output
        }                              # other more advanced options are available
                                       # as well..
        MyCtrlNode.__init__(self, name, terminals=terminals)
    def execute(self, growImage,seedImage=None, display=True):
        nh=4
        if self.ctrls['neighborhood'].currentIndex() == 1 :
            nn=8
        seg,numSeg = vigra.analysis.watersheds(image=growImage,neighborhood=4,seeds=seedImage)
        return {'dataOut': seg}

        
fclib.registerNodeType(Watershed ,[('Image-Segmentation',)])



class SmartWatershed(MyCtrlNode):
    """ (seeded) watershed"""
    nodeName = "SmartWatershed"

    uiTemplate=[

        ('Seed_NLD_et', 'spin', {'value': 0.25, 'step': 0.05, 'range': [0.01, None]}) ,
        ('Seed_NLD_scale',         'spin', {'value': 1.50, 'step': 0.25, 'range': [0.01, None]}) ,
        ('Seed_GM_sigma',          'spin', {'value': 1.0, 'step': 0.25, 'range': [0.01, None]}) ,
        ('Seed_GM_pow',            'spin', {'value': 2.0, 'step': 0.25, 'range': [0.01, 20]}) ,
        ('Seed_GM_S_sigma',        'spin', {'value': 0.7, 'step': 0.1, 'range': [0.01, 20]}) ,
        ('Seed_LM_neighborhood',   'combo', {'values': ['4', '8'], 'index': 1}),
        ('Seed_LI_neighborhood',   'combo', {'values': ['4', '8'], 'index': 1}),
        ('Grow_NLD_et', 'spin', {'value': 0.25, 'step': 0.05, 'range': [0.01, None]}) ,
        ('Grow_NLD_scale',         'spin', {'value': 1.50, 'step': 0.25, 'range': [0.01, None]}) ,
        ('Grow_GM_sigma',          'spin', {'value': 1.0, 'step': 0.25, 'range': [0.01, None]}) ,
        ('Grow_GM_pow',            'spin', {'value': 2.0, 'step': 0.25, 'range': [0.01, 20]}) ,
        ('Grow_GM_S_sigma',        'spin', {'value': 0.7, 'step': 0.1, 'range': [0.01, 20]}) ,
        ('Grow_neighborhood',      'combo', {'values': ['4', '8'], 'index': 0})
    ]
    def __init__(self, name):
        terminals = {
            'dataIn': dict(io='in'),  
            'dataOut': dict(io='out')  # to specify whether it is input or output
        }                              # other more advanced options are available
                                       # as well..
        MyCtrlNode.__init__(self, name, terminals=terminals)
    def execute(self,dataIn, display=True):

        ################################################
        #   SEEDING
        ###############################################
        # diffuse seeding image
        et    = self.ctrls['Seed_NLD_et'].value()
        scale = self.ctrls['Seed_NLD_scale'].value()
        seedImgDiff = vigra.filters.nonlinearDiffusion(dataIn,edgeThreshold=et,scale=scale)
        
        # compute gradient on seeding image
        sigma  = self.ctrls['Seed_GM_sigma'].value()
        powF   = self.ctrls['Seed_GM_pow'].value()
        sigmaS = self.ctrls['Seed_GM_S_sigma'].value()
        seedGradMag      = vigra.filters.gaussianGradientMagnitude(seedImgDiff,sigma=sigma)
        seedGradMagPow   = seedGradMag**powF
        seedGradSmoothed = vigra.filters.gaussianSmoothing(seedGradMagPow,sigma=sigmaS)

        # get seed image
        nhLM = convertNh(self.ctrls['Seed_LM_neighborhood'].currentIndex())
        nhLI = convertNh(self.ctrls['Seed_LM_neighborhood'].currentIndex())
        local_min=vigra.analysis.localMinima(seedGradSmoothed,neighborhood=nhLM)
        seeds=vigra.analysis.labelImageWithBackground(local_min,neighborhood=nhLI)


        ################################################
        #   evaluation map
        ###############################################
        # diffuse grow image
        et    = self.ctrls['Seed_NLD_et'].value()
        scale = self.ctrls['Seed_NLD_scale'].value()
        # do it
        growImgDiff = vigra.filters.nonlinearDiffusion(dataIn,edgeThreshold=et,scale=scale)
        
        # compute gradient on grow image
        sigma  = self.ctrls['Grow_GM_sigma'].value()
        powF   = self.ctrls['Grow_GM_pow'].value()
        sigmaS = self.ctrls['Grow_GM_S_sigma'].value()
        # do it
        growGradMag      = vigra.filters.gaussianGradientMagnitude(growImgDiff,sigma=sigma)
        growGradMagPow   = growGradMag**powF
        growGradSmoothed = vigra.filters.gaussianSmoothing(growGradMagPow,sigma=sigmaS)

        nh = convertNh(self.ctrls['Grow_neighborhood'].currentIndex())
        # watersheds
        labels,numseg=vigra.analysis.watersheds( 
            image        = growGradSmoothed,
            seeds        = seeds,
            neighborhood = nh,
            method       = 'RegionGrowing'
        )




        #nh=convertNh(self.ctrls['neighborhood'].currentIndex())

        #seg,numSeg = vigra.analysis.watersheds(image=growImage,neighborhood=4,seeds=seedImage)
        return {'dataOut': labels}

        
fclib.registerNodeType(SmartWatershed ,[('Image-Segmentation',)])




def nifty_sp(
    imgRGB,
    edgeThreshold    = 0.25,
    scale            = 20.0,
    sigmaGradMagSeed = 1.5,
    powSeedMap       = 2,
    sigmaSmooth      = 0.7,
    sigmaGradMagGrow = 1.2
):
    assert isinstance(imgRGB, vigra.VigraArray)
    img = vigra.colors.transform_RGB2Lab(imgRGB)
    assert isinstance(img, vigra.VigraArray)
    
    #print "diffuse"
    diffImg = vigra.filters.nonlinearDiffusion(img,edgeThreshold, scale)

    #print "smart watershed"
    # find seeds 
    #print "gaussianGradientMagnitude on diffImg=%r with sigma=%f" % (diffImg.shape, sigmaGradMagSeed)
    seeding_map  = vigra.filters.gaussianGradientMagnitude(diffImg,sigmaGradMagSeed)
    #print "seeding_map: shape=%r" % (seeding_map.shape,)
    seeding_map  = vigra.filters.gaussianSmoothing(seeding_map**powSeedMap,sigmaSmooth)
    local_minima = vigra.analysis.extendedLocalMinima(seeding_map)
    seed_map     = vigra.analysis.labelImageWithBackground(local_minima,neighborhood=8)
    #print "seed_map: %d labels" % seed_map.max()

    # evaluation map
    evaluation_map = vigra.filters.gaussianGradientMagnitude(diffImg,sigmaGradMagGrow)

    # watersheds
    labels,numseg=vigra.analysis.watersheds( 
        image        = evaluation_map,
        seeds        = seed_map,
        neighborhood = 4,
        method       = 'RegionGrowing'
    )


    #print "%d superpixels" % numseg

    #print "get init cgp and resample image"
    #print "numseg",numseg,labels.min(),labels.max()
    cgp,grid=superimg.cgpFromLabels(labels)

    #imgRGBBig = vigra.sampling.resize(img,cgp.shape,0)
    #superimg.visualize(imgRGBBig,cgp,np.ones(cgp.numCells(1),dtype=np.float32), cmap='jet',title='mixed')
   
    assert labels.shape[2] == 1
    labels = labels.squeeze()
    assert labels.ndim == 2, "labels has shape %r" % (labels.shape,)

    return labels,numseg


def _permuteLabels(data):

    flat = np.array(data).reshape([-1])
    unique , relabeling = np.unique(flat,return_inverse=True)
    permUnique = np.random.permutation(unique)
    flatNew = permUnique[relabeling]
    newLabels = flatNew.reshape([data.shape[0],data.shape[1]])
    return newLabels

node = numpyInNumpyOutNode(
    nodeName="PermuteLabels",
    uiTemplate=[
    ],
    f=_permuteLabels
)
fclib.registerNodeType(node,[('Image-Segmentation',)])

