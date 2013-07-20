ivigraph
========

ivigraph and interactive version of vigranumpy with a labView-ish gui for image processing operator  flow charts


Quickstart:
============
run flow.py and change the "lena" path to an image of your choise.
Cick on Flowchart-Button in the GUI and edit the graph 

- right click to add new operators 
- del to remove connections  unknown bug, this might lead to a floating point error for huge graphs
- connect operator to one of the four image viewers


Dependencies:
=============
- vigranumpy   http://hci.iwr.uni-heidelberg.de/vigra/  Python and Boost Python Wrapped C++ Code
    - one currently needs a very new version if vigra .
    - Use the newest version from https://github.com/ukoethe/vigra master
- pyqtgraph   http://www.pyqtgraph.org/   pure Python

Known Bugs:
=============

- load:
	- one might need change the histogram range a tiny bit to
	  refresh the image
- sometimes one gets floating point error when deleting edges no idea what is going on

Version
=============

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



Authors
=============
- 1 Thorsten Beier
- 2 Philipp Hanslovsky