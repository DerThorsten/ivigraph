ivigraph
========

ivigraph and interactive version of vigranumpy with a labView-ish gui for image processing operator  flow charts


Quickstart:
============
run flow.py and change the "lena" path to an image of your choise.
Cick on Flowchart-DockArea (flowchart itself might be hidden) in the GUI and edit the graph 

- right click to add new operators 
- press "del" to remove connections and nodes
- connect operator to one of the 4 image viewers


Dependencies:
=============
- numpy
- vigranumpy   http://hci.iwr.uni-heidelberg.de/vigra/  Python and Boost Python Wrapped C++ Code
    - one currently needs a very new version if vigra .
    - Use the newest version from https://github.com/ukoethe/vigra master
- pyqtgraph   http://www.pyqtgraph.org/   pure Python

- termcolor
    - sudo easy_install termcolor
- scipy



Known Bugs:
=============
- ColorChannel Nodes do not work on some machines ( strage vigra c++ signature error)
- Load:
	- one might need change the histogram range a tiny bit to
	  refresh the image
- Sometimes one gets floating point error when deleting edges no idea what is going on
    - (can someone reproduce this error?)


Authors
=============
- Thorsten Beier
- Philipp Hanslovsky


Version
=============
- 0.0.6.11
    - new node gui is intruduced in the first real node (numpy require node as first usecase)

- 0.0.6.10
    - fixed tensor nodes a bit ( some of them work with colorchannel images now)

- 0.0.6.9
    - batch mode image selector combo box is working

- 0.0.6.8
    - viewer will autoselect best mode for image 

- 0.0.6.7
    - experimental region features (dummy,does not work so far)
    - added make list , make tupe node

- 0.0.6.6
    - refactored node base classes 
    - split operators into seperated files
    - removed some unused imports

- 0.0.6.5
    - fixed minor bug in vigra_machine_learning.py

- 0.0.6.4
    - use MyNode and MyCtrlNode as base for allmost all operators

- 0.0.6.3
    - removed stupid prints
    - added colored printing (needs termcolor)
    - improved base class for ctrl nodes
    - added timing to "vigra" nodes
    - (hot-) fixed a so far unknown loading bug
        (error was raised but catched)

- 0.0.6.2
    - added a few demos

- 0.0.6.1
    - implemented a "batch-mode" to have a folder 
     with images as input .
    - image selection gui is enabled if input mode is batch mode

- 0.0.6.0
    - implemented a central widget ivigraph

- 0.0.5.9
    - added superpixel visualization
    - added a "demo" folder to repo where nice example flowcharts can be stored

- 0.0.5.8
    - removed unused buttons: (flowchart,reloadLibs,Norm)
    - added experimental gui for image selection (only GUI, no functionality so far)
    - refactored Viewer a bit
    - added label alpha slider 

- 0.0.5.7
    - added slicSuperpixels (and RegionToEdges for visualization)

- 0.0.5.6
    - added Random Forest classifier nodes (including auxiliary nodes)

- 0.0.5.5   
    - added numpy whereNotNode
    - started to make it easy to implement a generalize numpy.where
- 0.0.5.4   
    - added numpy require node
- 0.0.5.3   
    - readded  the functionality to switch between rgb,lab,and 1-channel 
    - removed unused ui templates from  viewNodes
- 0.0.5.2   
    - proof of concept for machine learning with random forest
- 0.0.5.1   
    - labeling uses multiple ImageItems in a layerd fashion
- 0.0.5.0   
    - INITAL HACKY VERSION OF SAVE AND LOAD IS WORKING
- 0.0.4.1   
    - added channel stacker with flexible number of input terminal
- 0.0.4.0   
    - changed main layout to a "dockarea"
- 0.0.3.5   
    - improved and fixed clear labels
- 0.0.3.4   
    - improved viewer gui
- 0.0.3.3   
    - extended imageview functions
- 0.0.3.2   
    - labeling is allmost finished for rgb color images
- 0.0.3.1   
    - fixed bug :brush size is now changeable
- 0.0.3.0   
    - proof of concept for labeling is done
- 0.0.2.0   
    - code works on differnt machines
- 0.0.1.0   
    - initial version

