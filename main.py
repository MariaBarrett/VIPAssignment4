import lktrack
import glob	

imnames = glob.glob("dudekface/*/*.pgm")
imnames = sorted(imnames)
print imnames
#create tracker object
lkt = lktrack.LKTracker(imnames)

#for im,ft in lkt.track():
#	print 'tracking %d features' %len(ft)

#detect in first frame, track in the remaining

lkt.detect_points()
lkt.draw()
for i in range(len(imnames)-1):
	lkt.track_points()
	lkt.draw()