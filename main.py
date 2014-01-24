from __future__ import division
import lktrack 
import glob
import pylab as plt
from PIL import Image


imnames = glob.glob("dudekface/*/*.pgm")
imnames = sorted(imnames,reverse=True)

print "Calculating."
#create tracker object
lkt = lktrack.LKTracker(imnames[:100])

ims = []
for im,ft in lkt.track():
	print 'tracking %d features' % len(ft)

# plot the tracks
plt.imshow(im) 
for p in ft:
	plt.plot(p[0],p[1],'bo')
for t in lkt.tracks:
<<<<<<< HEAD
	plt.plot([p[1] for p in t],[p[0] for p in t],'r-')
=======
	plt.plot([p[0] for p in t],[p[1] for p in t]) #switch 1 and 0 araound when running our implementation
>>>>>>> 7756b1f35d477c4ff521ab6655f5368db6d12cae
plt.axis('off')
plt.show()
