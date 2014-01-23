from __future__ import division
import cv2
import scipy.signal as si
import numpy.linalg as lin
from numpy import *
from scipy.ndimage import filters
from PIL import Image

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

	def harris(sigma=1.4,min_dist=10,threshold=0.03):
		""" From CV Draft. Compute the Harris corner detector response function
		for each pixel in a graylevel image. Return corners from a Harris response image
		min_dist is the minimum number of pixels separating corners and image boundary. """
		
		#self.image = cv2.imread(self.imnames[self.current_frame])
		self.image = array(Image.open(self.imnames[self.current_frame]).convert('L')) 

		# derivatives
		imx = zeros(im.shape)
		filters.gaussian_filter(im, (sigma,sigma), (0,1), imx) 
		imy = zeros(im.shape)
		filters.gaussian_filter(im, (sigma,sigma), (1,0), imy)

			# compute components of the Harris matrix
		Wxx = filters.gaussian_filter(imx*imx,sigma) 
		Wxy = filters.gaussian_filter(imx*imy,sigma) 
		Wyy = filters.gaussian_filter(imy*imy,sigma)
		# determinant and trace
		Wdet = Wxx*Wyy - Wxy**2
		Wtr = Wxx + Wyy
		harrisim = Wdet / Wtr

		# find top corner candidates above a threshold
		corner_threshold = harrisim.max() * threshold
		harrisim_t = (harrisim > corner_threshold) * 1

		# get coordinates of candidates
		coords = array(harrisim_t.nonzero()).T # ...and their values
		candidate_values = [harrisim[c[0],c[1]] for c in coords] 
		# sort candidates
		index = argsort(candidate_values)
			
		# store allowed point locations in array
		allowed_locations = zeros(harrisim.shape) 
		allowed_locations[min_dist:-min_dist,min_dist:-min_dist] = 1
			
		# select the best points taking min_distance into account
		filtered_coords = [] 
		for i in index:
			if allowed_locations[coords[i,0],coords[i,1]] == 1:
				filtered_coords.append(coords[i]) 
				allowed_locations[(coords[i,0]-min_dist):(coords[i,0]+min_dist),
					(coords[i,1]-min_dist):(coords[i,1]+min_dist)] = 0 
		self.features = filtered_coords


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

	"""
	def gauss_kern(self):
	   h1 = 15
	   h2 = 15
	   x, y = mgrid[0:h2, 0:h1]
	   x = x-h2/2
	   y = y-h1/2
	   sigma = 1.5
	   g = exp( -( x**2 + y**2 ) / (2*sigma**2) )
	   
	   return g / g.sum()

	def deriv(self,im1, im2):
	   g = self.gauss_kern()
	   Img_smooth = si.convolve(im1,g,mode='same')
	   fx,fy=gradient(Img_smooth)  
	   ft = si.convolve2d(im1, 0.25 * ones((2,2))) + \
	       si.convolve2d(im2, -0.25 * ones((2,2)))
	                 
	   fx = fx[0:fx.shape[0]-1, 0:fx.shape[1]-1]  
	   fy = fy[0:fy.shape[0]-1, 0:fy.shape[1]-1]
	   ft = ft[0:ft.shape[0]-1, 0:ft.shape[1]-1]
	   
	   return fx, fy, ft
	"""

	
	""" Here's a bet on how the bloody Lucas-Kanade can be written. Sorry for no comments.
	I just found it from here http://ascratchpad.blogspot.dk/2011/10/optical-flow-lucas-kanade-in-python.html
	I suspect he first computes what I would call the harris corner detection.
	 He does the gaussian filter manually and computes the harris values on the ENTIRE image and then identifies the relevant points in deriv/lk. 
	 If nothing else, we should be able to use the last part here and rewrite the first thing to harris corner from OpenCV
	  """
	def lk(self, im1, im2, i, j, window_size) :
		fx, fy, ft = self.deriv(im1, im2)
		halfWindow = np.floor(window_size/2)

		curFx = fx[i-halfWindow-1:i+halfWindow,
		          j-halfWindow-1:j+halfWindow]
		curFy = fy[i-halfWindow-1:i+halfWindow,
		          j-halfWindow-1:j+halfWindow]
		curFt = ft[i-halfWindow-1:i+halfWindow,
		          j-halfWindow-1:j+halfWindow]
		curFx = curFx.T
		curFy = curFy.T
		curFt = curFt.T

		curFx = curFx.flatten(order='F')
		curFy = curFy.flatten(order='F')
		curFt = -curFt.flatten(order='F')

		A = vstack((curFx, curFy)).T
		U = dot(dot(lin.pinv(dot(A.T,A)),A.T),curFt)

		return U[0], U[1]


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
				self.harris()
			else:
				self.track_points()

		#create a copy in RGB
		f = array(self.features).reshape(-1,2)
		im = cv2.cvtColor(self.image,cv2.COLOR_BGR2RGB)
		yield im,f