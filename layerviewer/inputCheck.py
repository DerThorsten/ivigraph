import numpy as np


class InputCheck(object):
	@staticmethod
	def colorImage(data):
		if data.ndim!=3 or data.shape[2]!=3:
			raise RuntimeError("Color Image needs ndim==3 and shape[2]=3")

	@staticmethod
	def grayImage(data):
		if data.ndim not in [2,3] or (data.ndim==3 and data.shape[2]!=1):
			raise RuntimeError("Gray Image needs ndim==2 (or ndim==3 and shape[2]==1)")


	@staticmethod
	def multiGrayImage(data):
		if data.ndim!=3 :
			raise RuntimeError("Multi Gray Image needs ndim==3")
