import numpy as np


class InputCheck(object):
	@staticmethod
	def colorImage(data):
		if data.ndim!=3 or data.shape[2]!=3:
			raise RuntimeError("Color Image needs ndim==3 and shape[2]=3")

	def grayImage(data):
		if data.ndim!=2 or (data.ndim==3 and data.shappe[2]!=3):
			raise RuntimeError("Gray Image needs ndim==2 (or ndim==3 and shape[2])")



	@staticmethod
	def multiGrayImage(data):
		if data.ndim!=3 :
			raise RuntimeError("Multi Gray Image needs ndim==3")
