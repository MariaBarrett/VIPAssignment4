from __future__ import division
import lktrack 
import glob
from pylab import *	
from PIL import Image


imnames = glob.glob("dudekface/*/*.pgm")
imnames = sorted(imnames)


#create tracker object
lkt = lktrack.LKTracker(imnames[:5])

for im,ft in lkt.track():
	print 'Tracking %d features' %len(ft)


#detect in first frame, track in the remaining

figure()
imshow(im)
for p in ft:
	plot(p[1],p[0],'bo')
for t in lkt.tracks:
	print t
	plot([p[0][1] for p in t],[p[0][0] for p in t])
axis('off')
show()
