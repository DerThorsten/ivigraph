ivigraph
========

ivigraph and interactive version of vigranumpy with a labView-ish gui for image processing operator  flow charts


Quickstart:
============
run flow.py (and change the "lena" path to an image of your choise).
Cick on Flowchart-Button in the GUI and edit the graph 

- right click to add new operators 
- del to remove connections ( unknown bug, this might lead to a floating point error for huge graphs)
- connect operator to one of the four image viewers


Dependencies:
=============
- vigranumpy   http://hci.iwr.uni-heidelberg.de/vigra/  (Python and Boost Python Wrapped C++ Code)
- pyqtgraph   http://www.pyqtgraph.org/   (pure Python)

Known Bugs:
=============

- load:
	- one might need change the histogram range a tiny bit to
	  refresh the image
- sometimes one gets floating point error when deleting edges (no idea what is going on)

Version
=============

- 0.0.5.3 - alpha  
	- readded  the functionality to switch between rgb,lab,and 1-channel 
	- removed unused ui templates from  viewNodes
- 0.0.5.2 - alpha  (proof of concept for machine learning with random forest)
- 0.0.5.1 - alpha  (labeling uses multiple ImageItems in a layerd fashion)
- 0.0.5.0 - alpha  (INITAL HACKY VERSION OF SAVE AND LOAD IS WORKING)
- 0.0.4.1 - alpha  (added channel stacker with flexible number of input terminal)
- 0.0.4.0 - alpha  (changed main layout to a "dockarea")
- 0.0.3.5 - alpha  (improved and fixed clear labels)
- 0.0.3.4 - alpha  (improved viewer gui)
- 0.0.3.3 - alpha  (extended imageview functions)
- 0.0.3.2 - alpha  (labeling is allmost finished for rgb color images)
- 0.0.3.1 - alpha  (fixed bug :(brush size is now changeable))
- 0.0.3.0 - alpha  (proof of concept for labeling is done)
- 0.0.2.0 - alpha  (code works on differnt machines)
- 0.0.1.0 - alpha  (initial version)



Authors
=============
- (1) Thorsten Beier
- (2) Philipp Hanslovsky