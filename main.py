import lktrack
import glob
from pylab import *	

imnames = glob.glob("dudekface/*/*.pgm")
imnames = sorted(imnames)

#create tracker object
lkt = lktrack.LKTracker(imnames[:25])

for im,ft in lkt.track():
	print 'tracking %d features' %len(ft)

#detect in first frame, track in the remaining

figure()
imshow(im)
for p in ft:
	plot(p[0],p[1],'bo')
for t in lkt.tracks:
	plot([p[0] for p in t],[p[1] for p in t])
axis('off')
show()