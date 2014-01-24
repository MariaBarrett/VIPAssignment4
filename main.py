from __future__ import division
import lktrack 
import glob
import pylab as plt
from PIL import Image


imnames = glob.glob("dudekface/*/*.pgm")
imnames = sorted(imnames,reverse=True)

print "Calculating."
#create tracker object
lkt = lktrack.LKTracker(imnames[:25])

ims = []
for im,ft in lkt.track():
	print 'Tracking %d features' %len(ft)


#detect in first frame, track in the remaining

plt.figure()
plt.imshow(im)
for p in ft:
	plt.plot(p[1],p[0],'bo')
for t in lkt.tracks:
	plt.plot([p[1] for p in t],[p[0] for p in t])

plt.axis('off')
plt.show()
