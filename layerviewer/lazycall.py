
def lazyCall(value,f,*args,**kwargs):
	if value is None:
		return f(*args,**kwargs)
	else:
		return value