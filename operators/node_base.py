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




class Found(Exception): pass

class AdvCtrlNode(Node):
    """Abstract class for nodes with auto-generated control UI"""
    
    sigStateChanged = QtCore.Signal(object)
    def __init__(self, name, ui=None, terminals=None):

        Node.__init__(self, name=name, terminals=terminals)
        
        

        #self.widget = QtGui.QWidget()
        #l = QtGui.QVBoxLayout()
        #l.setSpacing(0)
        #self.widget.setLayout(l)


        self.param = Parameter.create(name='params', type='group', children=ui)
            
        self.uiTree = ParameterTree()
        self.uiTree.setParameters(self.param , showTop=False)


        #l.addWidget(self.uiTree)


        self.param.sigTreeStateChanged.connect(self.changedTree)
    
    def ctrlWidget(self):
        return self.uiTree
       
    def changedTree(self,param,changes):

        print("tree changes:")
        for param, change, data in changes:
            path = self.param.childPath(param)
            if path is not None:
                childName = '.'.join(path)
            else:
                childName = param.name()
            print('  parameter: %s'% childName)
            print('  change:    %s'% change)
            print('  data:      %s'% str(data))
            print('  ----------')
        self.changed()
        

    def onAction(self,actionName):
        pass

    def changed(self):  
        self.update()
        self.sigStateChanged.emit(self)

    def startProcess(self):
        print colored('\nStart', 'blue'), colored(self.name(), 'green')
        self.t0 = time.time()
    def endProcess(self):
        t1 = time.time()
        t = t1-self.t0
        print colored('Done ', 'blue'),colored("%f sek"%float(t),'green')

    """
    def graphicsItem(self):
        if self._graphicsItem is None:
            self._graphicsItem = CustomNodeGraphicsItem(self,(200,200))
        return self._graphicsItem
    """

    def execute(self, *args, **kwargs):
        pass


    def process(self, *args, **kwargs):
        self.startProcess()
        return_value = self.execute(*args, **kwargs)
        self.endProcess()
        return return_value


    def getParamValue(self,names):
        depth = len(names)
        print "get values "



        if depth == 1 : 
            return self.param.getValues()[names[0]][0]

        if depth >= 2 :


            current  = None
            currentP = None

            cp  = self.param

            print "startname ",cp.name()

            counter=0
            while(True):
                #print "while loop enter name ",cp.name(),counter


                if cp.name()==names[len(names)-1]:
                    #print "\n\n\nLAST NAME MATCH"
                    pass
                try :
                    for d,name in enumerate(names[counter:len(names)]):
                        doBreak=False
                        #print "d,name",d,name
                        childs =  cp.children()
                        #print " \n ITERATE CHILDS \n"
                        for ci,c in enumerate(childs) :

                            cName = c.name()
                            #print "cname",cName
                            if cName == name :
                                #print "names match"
                                
                                cpOld=cp
                                cp=c
                               


                                #print "____counter",counter
                                #print "____current",cp
                               #print "____currentPara",cp.getValues()

                                if counter == depth -1:
                                    #print "in depth -1  ci",ci
                                    #print "try in old" ,type(cpOld.getValues())," len ",len(cpOld.getValues())
                                    #print "try in old2" ,cpOld.getValues()[cName][0]
                                    return cpOld.getValues()[cName][0]
                   
                                counter+=1

                                raise Found

                            else :
                                #print "names do not match"
                                continue

                            #print "name", c.name()
                            #print "get values",c.getValues()
                
                except Found:
                    #print "\n\n\n FOund"
                    pass

                #print "counter",counter
                #print "current",cp
                #print "currentPara",cp.getValues()
                if counter == depth:
                    #print "finished??!?"
                    break

               



            return current[0]


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


def numpyInNumpyOutNode(nodeName,uiTemplate,f,dtypeIn=np.float32,dtypeOut=np.float32,doChannelWise=False,nodeSize=(100,100)):
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
            else:

                numChannels = dataInVigra.shape[2]
                vigraResult  = np.ones( dataInVigra.shape,dtype=dtypeIn)
                for c in range(numChannels):
                    #print "channel wise input :",dataInVigra[:,:,c].shape,dataInVigra[:,:,c].dtype
                    vigraResult[:,:,c]=f(dataInVigra[:,:,c],**kwargs)
            vigraResult = np.squeeze(vigraResult)
            vigraResult = np.require(vigraResult,dtype=dtypeOut)

            return {'dataOut': vigraResult}

    return _numpyInNumpyOutNodeImpl        







