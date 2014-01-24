from __future__ import division
import cv2
import scipy.signal as si
import numpy.linalg as lin
from numpy import *
from scipy.ndimage import filters
from PIL import Image

#some constants and default parameters


"""Class for Lucas-Kanade tracking with pyramidal optical flow.
	'Skeleton' taken from CV Draft, but own LK implementation. 
"""
class LKTracker(object):

	""" Initialize with a list of image names """
	def __init__(self,imnames):
		self.imnames = imnames
		self.features = [] #corner points
		self.tracks = [] #obviously the tracked features
		self.current_frame = 0
		self.sigma = 3


	def harris(self,min_dist=8,threshold=0.05):
		""" Compute the Harris corner detector response function
		for each pixel in a graylevel image. Return corners from a Harris response image
		min_dist is the minimum number of pixels separating corners and image boundary. """

		self.image = cv2.imread(self.imnames[self.current_frame])
		self.gray = cv2.cvtColor(self.image,cv2.COLOR_BGR2GRAY)

		# derivatives
		imx = zeros(self.gray.shape)
		filters.gaussian_filter(self.gray, (self.sigma,self.sigma), (0,1), imx) 
		imy = zeros(self.gray.shape)
		filters.gaussian_filter(self.gray, (self.sigma,self.sigma), (1,0), imy)

			# compute components of the Harris matrix
		Wxx = filters.gaussian_filter(imx*imx,self.sigma) 
		Wxy = filters.gaussian_filter(imx*imy,self.sigma) 
		Wyy = filters.gaussian_filter(imy*imy,self.sigma)
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

		filtered_coords = array(filtered_coords)

		self.features = filtered_coords
		self.tracks = [[p] for p in filtered_coords.reshape((-1,2))]
		self.prev_gray = self.gray



	"""Here we track the detected features. Surprising eh?
	 We utilize the found features and try to calculate the OpticalFlow with the Lucas-Kanade method"""
	def track_points(self):
		if self.features != []:
			self.step() #move to the next frame - surprisign too eh!

		#load the images and create grayscale
		self.image = cv2.imread(self.imnames[self.current_frame])
		self.gray = cv2.cvtColor(self.image,cv2.COLOR_BGR2GRAY)

		#reshape to fit input format
		tmp = float32(self.features).reshape(-1, 1, 2)
		tmpf = []

		#print tmp
		#print type(tmp)
		tmpf=[]
		ims1 = filters.gaussian_filter(self.prev_gray,self.sigma)
		ims2 = filters.gaussian_filter(self.gray,self.sigma)
		for elem in tmp:
			inner = []
			inner = self.lk(ims1,ims2,elem[0][0],elem[0][1],15)
			tmpf.append(inner)

		for i in range(len(tmpf)):
			self.features[i][0] = self.features[i][0]+tmpf[i][0]
			self.features[i][1] = self.features[i][1]+tmpf[i][1]
		
		#clean tracks from lost points
		self.prev_gray = self.gray


	""" Here we do the necessary derivations as to satisfy the Harris matrix later on. 
	"""
	def deriv(self,im1, im2):
	   """
	   g = self.gauss_kern()
	   Img_smooth = si.convolve(im1,g,mode='same')
	   """
	   fx,fy=gradient(im1)  
	   ft = si.convolve2d(im1, 0.25 * ones((2,2))) + si.convolve2d(im2, -0.25 * ones((2,2)))
	                 
	   fx = fx[0:fx.shape[0]-1, 0:fx.shape[1]-1]  
	   fy = fy[0:fy.shape[0]-1, 0:fy.shape[1]-1]
	   ft = ft[0:ft.shape[0]-1, 0:ft.shape[1]-1]
	   
	   return fx, fy, ft
	
	
	""" Here's a bet on how the bloody Lucas-Kanade can be written. 
	"""
	def lk(self, im1, im2, i, j, window_size):
		fx, fy, ft = self.deriv(im1, im2)
		hwin = floor(window_size/2)

		Fx = fx[i-hwin-1:i+hwin,
		          j-hwin-1:j+hwin]
		Fy = fy[i-hwin-1:i+hwin,
		          j-hwin-1:j+hwin]
		Ft = ft[i-hwin-1:i+hwin,
		          j-hwin-1:j+hwin]
		Fx = Fx.T
		Fy = Fy.T
		Ft = Ft.T

		Fx = Fx.flatten(order='F')
		Fy = Fy.flatten(order='F')
		Ft = -Ft.flatten(order='F')

		A = vstack((Fx, Fy)).T
		U = dot(dot(lin.pinv(dot(A.T,A)),A.T),Ft)

		return U[0], U[1]


	"""Step to another frame. If no argument is given, step to the next frame. """
	def step(self,framenbr=None):
		if framenbr is None:
			self.current_frame = (self.current_frame +1) % len(self.imnames)
		else:
			self.current_frame = framenbr % len(self.imnames)


	"""Drawing with CV itself. Not sure why this is so smart, but whatevs. Press any key to continue to next frame """
	def draw(self):

		#draw points as green circles
		for point in self.features:
			cv2.circle(self.image,(int(point[0][0]),int(point[0][1])),3,(0,255,0),-1)

		cv2.imshow('LKtrack',self.image[0])
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