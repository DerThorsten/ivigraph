from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg
import pyqtgraph.functions as fn
from pyqtgraph.graphicsItems.GraphicsObject import GraphicsObject

class CustomNodeGraphicsItem(pg.GraphicsObject):
    def __init__(self, node,size=(100,100)):
        #QtGui.QGraphicsItem.__init__(self)
        pg.GraphicsObject.__init__(self)
        #QObjectWorkaround.__init__(self)
        
        #self.shadow = QtGui.QGraphicsDropShadowEffect()
        #self.shadow.setOffset(5,5)
        #self.shadow.setBlurRadius(10)
        #self.setGraphicsEffect(self.shadow)
        
        self.pen = fn.mkPen(0,0,0)
        self.selectPen = fn.mkPen(200,200,200,width=2)
        self.brush = fn.mkBrush(200, 200, 200, 150)
        self.hoverBrush = fn.mkBrush(200, 200, 200, 200)
        self.selectBrush = fn.mkBrush(200, 200, 255, 200)
        self.hovered = False
        
        self.node = node
        flags = self.ItemIsMovable | self.ItemIsSelectable | self.ItemIsFocusable |self.ItemSendsGeometryChanges
        #flags =  self.ItemIsFocusable |self.ItemSendsGeometryChanges

        self.setFlags(flags)
        self.bounds = QtCore.QRectF(0, 0, size[0], size[1])
        self.nameItem = QtGui.QGraphicsTextItem(self.node.name(), self)
        self.nameItem.setDefaultTextColor(QtGui.QColor(50, 50, 50))
        self.nameItem.moveBy(self.bounds.width()/2. - self.nameItem.boundingRect().width()/2., 0)
        self.nameItem.setTextInteractionFlags(QtCore.Qt.TextEditorInteraction)
        self.updateTerminals()
        #self.setZValue(10)

        self.nameItem.focusOutEvent = self.labelFocusOut
        self.nameItem.keyPressEvent = self.labelKeyPress
        
        self.menu = None
        self.buildMenu()
        
        self.inProcess=False
        #self.node.sigTerminalRenamed.connect(self.updateActionMenu)
        
    #def setZValue(self, z):
        #for t, item in self.terminals.itervalues():
            #item.setZValue(z+1)
        #GraphicsObject.setZValue(self, z)
        
    def labelFocusOut(self, ev):
        QtGui.QGraphicsTextItem.focusOutEvent(self.nameItem, ev)
        self.labelChanged()
        
    def labelKeyPress(self, ev):
        if ev.key() == QtCore.Qt.Key_Enter or ev.key() == QtCore.Qt.Key_Return:
            self.labelChanged()
        else:
            QtGui.QGraphicsTextItem.keyPressEvent(self.nameItem, ev)
        
    def labelChanged(self):
        newName = str(self.nameItem.toPlainText())
        if newName != self.node.name():
            self.node.rename(newName)
            
        ### re-center the label
        bounds = self.boundingRect()
        self.nameItem.setPos(bounds.width()/2. - self.nameItem.boundingRect().width()/2., 0)

    def setPen(self, pen):
        self.pen = pen
        self.update()
        
    def setBrush(self, brush):
        self.brush = brush
        self.update()
        
        
    def updateTerminals(self):
        bounds = self.bounds
        self.terminals = {}
        inp = self.node.inputs()
        dy = bounds.height() / (len(inp)+1)
        y = dy
        for i, t in inp.items():
            item = t.graphicsItem()
            item.setParentItem(self)
            #item.setZValue(self.zValue()+1)
            br = self.bounds
            item.setAnchor(0, y)
            self.terminals[i] = (t, item)
            y += dy
        
        out = self.node.outputs()
        dy = bounds.height() / (len(out)+1)
        y = dy
        for i, t in out.items():
            item = t.graphicsItem()
            item.setParentItem(self)
            item.setZValue(self.zValue())
            br = self.bounds
            item.setAnchor(bounds.width(), y)
            self.terminals[i] = (t, item)
            y += dy
        
        #self.buildMenu()
        
        
    def boundingRect(self):
        return self.bounds.adjusted(-5, -5, 5, 5)
        
        
    def mousePressEvent(self, ev):
        ev.ignore()


    def mouseClickEvent(self, ev):
        #print "Node.mouseClickEvent called."
        if int(ev.button()) == int(QtCore.Qt.LeftButton):
            ev.accept()
            #print "    ev.button: left"
            sel = self.isSelected()
            #ret = QtGui.QGraphicsItem.mousePressEvent(self, ev)
            self.setSelected(True)
            if not sel and self.isSelected():
                #self.setBrush(QtGui.QBrush(QtGui.QColor(200, 200, 255)))
                #self.emit(QtCore.SIGNAL('selected'))
                #self.scene().selectionChanged.emit() ## for some reason this doesn't seem to be happening automatically
                self.update()
            #return ret
        
        elif int(ev.button()) == int(QtCore.Qt.RightButton):
            #print "    ev.button: right"
            ev.accept()
            #pos = ev.screenPos()
            self.raiseContextMenu(ev)
            #self.menu.popup(QtCore.QPoint(pos.x(), pos.y()))
            
    def mouseDragEvent(self, ev):
        #print "Node.mouseDrag"
        if ev.button() == QtCore.Qt.LeftButton:
            ev.accept()
            self.setPos(self.pos()+self.mapToParent(ev.pos())-self.mapToParent(ev.lastPos()))
        
    def hoverEvent(self, ev):
        if not ev.isExit() and ev.acceptClicks(QtCore.Qt.LeftButton):
            ev.acceptDrags(QtCore.Qt.LeftButton)
            self.hovered = True
        else:
            self.hovered = False
        self.update()
            
    def keyPressEvent(self, ev):
        if ev.key() == QtCore.Qt.Key_Delete or ev.key() == QtCore.Qt.Key_Backspace:
            ev.accept()
            if not self.node._allowRemove:
                return
            self.node.close()
        else:
            ev.ignore()

    def itemChange(self, change, val):
        if change == self.ItemPositionHasChanged:
            for k, t in self.terminals.items():
                t[1].nodeMoved()
        return GraphicsObject.itemChange(self, change, val)
            

    def getMenu(self):
        return self.menu

    def getContextMenus(self, event):
        return [self.menu]
    
    def raiseContextMenu(self, ev):
        menu = self.scene().addParentContextMenus(self, self.getMenu(), ev)
        pos = ev.screenPos()
        menu.popup(QtCore.QPoint(pos.x(), pos.y()))
        
    def buildMenu(self):
        self.menu = QtGui.QMenu()
        self.menu.setTitle("Node")
        a = self.menu.addAction("Add input", self.addInputFromMenu)
        if not self.node._allowAddInput:
            a.setEnabled(False)
        a = self.menu.addAction("Add output", self.addOutputFromMenu)
        if not self.node._allowAddOutput:
            a.setEnabled(False)
        a = self.menu.addAction("Remove node", self.node.close)
        if not self.node._allowRemove:
            a.setEnabled(False)
        
    def addInputFromMenu(self):  ## called when add input is clicked in context menu
        self.node.addInput(renamable=True, removable=True, multiable=True)
        
    def addOutputFromMenu(self):  ## called when add output is clicked in context menu
        self.node.addOutput(renamable=True, removable=True, multiable=False)
        

    def paint(self, p, *args):
        p.setPen(self.pen)

        if self.isSelected():
            p.setPen(self.selectPen)
            p.setBrush(self.selectBrush)
        else:
            p.setPen(self.pen)
            if self.hovered:
                p.setBrush(self.hoverBrush)
            else:
                p.setBrush(self.brush)
        p.drawRect(self.bounds)

        