from __future__ import division
import lktrack 
import glob
from pylab import *	

imnames = glob.glob("dudekface/*/*.pgm")
imnames = sorted(imnames)
gray()


#create tracker object
lkt = lktrack.LKTracker(imnames[:25])

for im,ft in lkt.track():
	print 'Tracking %d features' %len(ft)
	

#im = array(Image.open(imnames[0]).convert('L')) 
#filtered_coords = lktrack.harris(im) 
#harris.plot_harris_points(im, filtered_coords)

""" Plots corners found in image. """
#figure()
#imshow(im)
#plot([p[1] for p in filtered_coords],[p[0] for p in filtered_coords],'*') 
#axis('off')
#show()


#detect in first frame, track in the remaining

figure()
imshow(im)
for p in ft:
	plot(p[0],p[1],'bo')
for t in lkt.tracks:
	plot([p[0] for p in t],[p[1] for p in t])
axis('off')
show()