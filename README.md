ivigraph
========

ivigraph and interactive version of vigranumpy with a labView-ish gui for image processing operator  flow charts

Version
=============
0.01 - alpha

Quickstart:
============
run flow.py (and change the "lena" path to an image of your choise).
Cick on Flowchart-Button in the GUI and edit the graph

Dependencies:
=============
- vigranumpy   http://hci.iwr.uni-heidelberg.de/vigra/  (Python and Boost Python Wrapped C++ Code)
- pyqtgraph   http://www.pyqtgraph.org/   (pure Python)

Known Bugs:
=============

- save does not work with the default pyqtgraph version
- load does not work with the patched version of pyqtgraph
- floating point error when deleting edges (no idea what is going on)

Authors
=============
- Thorsten Beier