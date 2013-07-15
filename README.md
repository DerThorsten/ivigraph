ivigraph
========

ivigraph and interactive version of vigranumpy with a labView-ish gui for image processing operator  flow charts

Version 0.030 alpha
=============

0.030 - alpha 	( proof of concept for labeling is done)
0.020 - alpha 	( code works on differnt machines)
0.010 - alpha  	( initial version)


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

- save does not work with the default pyqtgraph version
- load does not work with the patched version of pyqtgraph
- sometimes one gets floating point error when deleting edges (no idea what is going on)

Authors
=============
- Thorsten Beier
- Philipp Hanslovsky