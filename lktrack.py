import cv2
from numpy import *

#some constants and default parameters
lk_params = dict(
	winSize=(15,15),
	maxLevel=2,
	criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,10,0.03)) #He gives no reason for this - lets test it

subpix_params = dict(
	zeroZone=(-1,-1),
	winSize=(10,10),
	criteria = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS,20,0.03))

feature_params = dict(maxCorners=500,qualityLevel=0.01,minDistance=10)



"""Class for Lucas-Kanade tracking with pyramidal optical flow.
	All taken from the CV draft. 
"""
class LKTracker(object):

	""" Initialize with a list of image names """
	def __init__(self,imnames):
		self.imnames = imnames
		self.features = [] #corner points
		self.tracks = [] #obviously the tracked features
		self.current_frame = 0



	""" Detect 'good features to track' (corners) in the current frame
	using sub-pixel accuracy. """
	def detect_points(self):
		self.image = cv2.imread(self.imnames[self.current_frame])
		self.gray = cv2.cvtColor(self.image,cv2.COLOR_BGR2GRAY)

		#search for good points
		features = cv2.goodFeaturesToTrack(self.gray,**feature_params)

		#refine the corner locations - 'arris Corner Detection?
		cv2.cornerSubPix(self.gray,features, **subpix_params)

		self.features = features
		self.tracks = [[p] for p in features.reshape((-1,2))]

		self.prev_gray = self.gray



	"""Here we track the detected features. Surprising eh?
	 However, we should rewrite this into our own Lucas-Kanade"""
	def track_points(self):
		if self.features != []:
			self.step() #move to the next frame - surprisign too eh!

		#load the images and create grayscale
		self.image = cv2.imread(self.imnames[self.current_frame])
		self.gray = cv2.cvtColor(self.image,cv2.COLOR_BGR2GRAY)

		#reshape to fit input format
		tmp = float32(self.features).reshape(-1, 1, 2)

		#calculate optical flow
		features,status,track_error = cv2.calcOpticalFlowPyrLK(self.prev_gray,self.gray,tmp,None, **lk_params) 

		#remove points lost
		self.features = [p for (st,p) in zip(status,features) if st]

		#clean tracks from lost points
		features = array(features).reshape((-1,2))
		for i,f in enumerate(features):
			self.tracks[i].append(f)
		ndx = [i for (i,st) in enumerate(status) if not st]
		ndx.reverse() #remove from back
		for i in ndx:
			self.tracks.pop(i)

		self.prev_gray = self.gray



		"""Step to another frame. If no argument is given, step to the next frame. """
	def step(self,framenbr=None):
		if framenbr is None:
			self.current_frame = (self.current_frame +1) % len(self.imnames)
		else:
			self.current_frame = framenbr % len(self.imnames)


	"""Drawing with CV itself. Not sure why this is so smart, but whatevs. Press any key to close window """
	def draw(self):

		#draw points as green circles
		for point in self.features:
			cv2.circle(self.image,(int(point[0][0]),int(point[0][1])),3,(0,255,0),-1)

		cv2.imshow('LKtrack',self.image)
		cv2.waitKey()


	def track(self):
		"""Generator for stepping through a sequence. """

		for i in range(len(self.imnames)):
			if self.features == []:
				self.detect_points()
			else:
				self.track_points()

		#create a copy in RGB
		f = array(self.features).reshape(-1,2)
		im = cv2.cvtColor(self.image,cv2.COLOR_BGR2RGB)
		yield im,f