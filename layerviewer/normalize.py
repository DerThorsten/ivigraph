def norm01(dataIn,channelWise=False):
	out=dataIn.copy()
	if channelWise==False:
		out-=out.min()
		out/=out.max()
	else :
		for c in range(dataIn.shape[2]):
			out[:,:,c]-=out[:,:,c].min()
			out[:,:,c]/=out[:,:,c].max()
	return out


