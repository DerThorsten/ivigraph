from pyqtgraph.flowchart import Flowchart, Node
import pyqtgraph.flowchart.library as fclib
from pyqtgraph.flowchart.library.common import CtrlNode
from pyqtgraph.flowchart import Node
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg
import numpy as np
import scipy.ndimage
import vigra
import math
import functools
from scipy.stats.mstats import mquantiles
from collections import OrderedDict
from nodegraphics import CustomNodeGraphicsItem

from termcolor import colored
import time

import pyqtgraph.parametertree.parameterTypes as pTypes
from pyqtgraph.parametertree import Parameter, ParameterTree, ParameterItem, registerParameterType



class Opts(object):
    @staticmethod
    def dtypeOpt(optName='dtype',default=np.float32):
        dtypeOpt = {'name': optName, 'type': 'list', 'values': 
            {
                    "float32" : np.float32,
                    "float64" : np.float64,
                    "uint8" : np.uint8,
                    "uint16" : np.uint16,
                    "uint32" : np.uint32,
                    "uint64" : np.uint64,
                    "int8"   : np.int8,
                    "int16" : np.int16,
                    "int32" : np.int32,
                    "int64" : np.int64,
                    "bool" : np.bool
            }, 'value': default
        }
        return dtypeOpt

    @staticmethod
    def colormap(optName='colormap'):
        return {'name': optName, 'type': 'colormap','autoIncrementName':True}



class Found(Exception): pass

class AdvCtrlNode(Node):
    """Abstract class for nodes with auto-generated control UI"""
    
    sigStateChanged = QtCore.Signal(object)
    def __init__(self, name, userUi, terminals=None,size=(100,100),allowAddInput=False):
        self.size=size
        
        
        

        self.nodeOpt = {'name': 'NodeOptions', 'type': 'group','expanded':False, 'children': userUi}

        updateOpts = {'name': 'UpdateOptions','expanded':False, 'type': 'group', 'children': [   

            {'name': 'autoUpdate',       'type': 'bool',     'value':  True},
            {'name': 'update', 'type': 'action'},
        ]}



        ui = [self.nodeOpt,updateOpts]
        


        self.param = Parameter.create(name='params', type='group', children=ui)
            
        self.uiTree = ParameterTree()
        self.uiTree.setParameters(self.param , showTop=False)



        self.param.sigTreeStateChanged.connect(self.changedTree)
        Node.__init__(self, name=name, terminals=terminals,allowAddInput=allowAddInput)
    def ctrlWidget(self):
        return self.uiTree
       
    def changedTree(self,param,changes):
        """
        if *anything* changes in parameter tree
        this method will be called
        """

        autoUpdate = self.getParamValue('UpdateOptions','autoUpdate')
        #print("tree changes:",len(changes))
        nChanges = len(changes)
        for param, change, data in changes:
            path = self.param.childPath(param)

            if nChanges ==1 and path == ['UpdateOptions','autoUpdate'] :
                return


            if path==['UpdateOptions','update']:
                self.onUpdateOptionsUpdatePress()
                return

            #print('path',path)
            #print('  change:    %s'% change)
            if change == 'activated':
                if nChanges != 1 : 
                    raise RuntimeError('internal error')
                self.onActionButtonPress(path)
                return
        if autoUpdate:
            self.changed()
    
    def onUpdateOptionsUpdatePress(self):
        self.changed()

    def onActionButtonPress(self,actionPath):
        print "onActionButtonPress",actionPath

    def changed(self):  
        #print "changed !"
        self.update()
        self.sigStateChanged.emit(self)

    def startProcess(self):
        print colored('\nStart', 'blue'), colored(self.name(), 'green')
        self.t0 = time.time()
    def endProcess(self):
        t1 = time.time()
        t = t1-self.t0
        print colored('Done ', 'blue'),colored("%f sek"%float(t),'green')


    def graphicsItem(self):
        if self._graphicsItem is None:
            self._graphicsItem = CustomNodeGraphicsItem(self,self.size)
        return self._graphicsItem


    def execute(self, *args, **kwargs):
        pass


    def process(self, *args, **kwargs):
        self.startProcess()
        return_value = self.execute(*args, **kwargs)
        self.endProcess()
        return return_value


    def getParamValue(self,*names):
        #print "childs"
        #for c in self.param.children():
        #    print c.name()


        param=self.param.param(*names)
        return param.value()

    """
    def saveState(self):
        state = Node.saveState(self)
        state['ctrl'] = self.stateGroup.state()
        return state
    
    def restoreState(self, state):
        Node.restoreState(self, state)
        if self.stateGroup is not None:
            self.stateGroup.setState(state.get('ctrl', {}))
            
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


    """



def convertNh(nh):
    if nh == 0 : neighborhood = 4
    else: neighborhood =  8
    return neighborhood



class MyNode(Node):
    def __init__(self,name,terminals,nodeSize=(100,100),allowAddInput=False):
        self.nodeSize=nodeSize
        Node.__init__(self, name=name, terminals=terminals,allowAddInput=allowAddInput)   
        self.t0 = None

    def startProcess(self):
        print colored('\nStart', 'blue'), colored(self.name(), 'green')
        self.t0 = time.time()
    def endProcess(self):
        t1 = time.time()
        t = t1-self.t0
        print colored('Done ', 'blue'),colored("%f sek"%float(t),'green')


    def graphicsItem(self):
        if self._graphicsItem is None:
            self._graphicsItem = CustomNodeGraphicsItem(self,self.nodeSize)
        return self._graphicsItem


    def execute(self, *args, **kwargs):
        pass


    def process(self, *args, **kwargs):
        self.startProcess()
        return_value = self.execute(*args, **kwargs)
        self.endProcess()
        return return_value



class MyCtrlNode(CtrlNode):
    def __init__(self,name,terminals,nodeSize=(100,100)):
        self.nodeSize=nodeSize
        CtrlNode.__init__(self, name=name, terminals=terminals)   
        self.t0 = None

    def startProcess(self):
        print colored('\nStart', 'blue'), colored(self.name(), 'green')
        self.t0 = time.time()
    def endProcess(self):
        t1 = time.time()
        t = t1-self.t0
        print colored('Done ', 'blue'),colored("%f sek"%float(t),'green')

    def graphicsItem(self):
        if self._graphicsItem is None:
            self._graphicsItem = CustomNodeGraphicsItem(self,self.nodeSize)
        return self._graphicsItem


    def execute(self, *args, **kwargs):
        pass


    def process(self, *args, **kwargs):
        self.startProcess()
        return_value = self.execute(*args, **kwargs)
        self.endProcess()
        return return_value


def numpyInNumpyOutNode(nodeName,uiTemplate,f,dtypeIn=np.float32,dtypeOut=np.float32,doChannelWise=False,nodeSize=(100,100),tensor=False):
    name = nodeName
    uiT  = uiTemplate
    kwargs = {}
    for uit in uiTemplate:
        kwargs[uit[0]]=None



    class _numpyInNumpyOutNodeImpl(MyCtrlNode):
        """%s"""%name
        nodeName = name
        uiTemplate = uiT
        def __init__(self, name):
            ## Define the input / output terminals available on this node
            terminals = {
                'dataIn': dict(io='in'),    # each terminal needs at least a name and
                'dataOut': dict(io='out'),  # to specify whether it is input or output
            }                              # other more advanced options are available
                                           # as well..
            
            super(_numpyInNumpyOutNodeImpl,self).__init__( name=name, terminals=terminals,nodeSize=nodeSize)
            
            self.inProcess = False



        def execute(self, dataIn, display=True):

            if dataIn is None :
                assert False
            # CtrlNode has created self.ctrls, which is a dict containing {ctrlName: widget}
            for uit in uiTemplate:
                pName = uit[0]
                pType = uit[1]

                #print "name ",pName ,"pType",pType
                if(pType=='spin' or pType == 'intSpin'):
                    kwargs[pName]=self.ctrls[pName].value()
                elif(pType=='check'):
                    kwargs[pName]=self.ctrls[pName].isChecked()
                elif(pType=='combo'):
                    kwargs[pName]=self.ctrls[pName].currentIndex()


            #sigma = self.ctrls['sigma'].value()
            dataInVigra = np.require(dataIn,dtype=dtypeIn)
            dataInVigra = np.squeeze(dataInVigra)
            


            if doChannelWise == False or dataInVigra.ndim==2 or dataInVigra.shape[2]==1:
                #print "Single Input  ",dataInVigra.shape,dataInVigra.dtype
                vigraResult = f(dataInVigra,**kwargs)
            elif tensor == False:
                numChannels = dataInVigra.shape[2]
                vigraResult  = np.ones( dataInVigra.shape,dtype=dtypeIn)
                for c in range(numChannels):
                    #print "channel wise input :",dataInVigra[:,:,c].shape,dataInVigra[:,:,c].dtype
                    vigraResult[:,:,c]=f(dataInVigra[:,:,c],**kwargs)
            else :
                numChannels = dataInVigra.shape[2]
                r  = []
                for c in range(numChannels):
                    #print "channel wise input :",dataInVigra[:,:,c].shape,dataInVigra[:,:,c].dtype
                    r.append( f(dataInVigra[:,:,c],**kwargs) )

                vigraResult=np.concatenate(r,axis=2)



            vigraResult = np.squeeze(vigraResult)
            vigraResult = np.require(vigraResult,dtype=dtypeOut)

            return {'dataOut': vigraResult}

    return _numpyInNumpyOutNodeImpl        







