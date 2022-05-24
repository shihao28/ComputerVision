# import the necessary packages
from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np


class CentroidTracker:
	"""
	Referenced from PyImageSearch
	"""
	def __init__(self, maxDisappeared=100, maxDistance=100, bottom=True, n_init=1):
		# initialize the next unique object ID along with two ordered
		# dictionaries used to keep track of mapping a given object
		# ID to its centroid and number of consecutive frames it has
		# been marked as "disappeared", respectively
		self.nextObjectID = 0
		self.objects = OrderedDict()
		self.disappeared = OrderedDict()

		# store the number of maximum consecutive frames a given
		# object is allowed to be marked as "disappeared" until we
		# need to deregister the object from tracking
		self.maxDisappeared = maxDisappeared

		# store the maximum distance between centroids to associate
		# an object -- if the distance is larger than this maximum
		# distance we'll start to mark the object as "disappeared"
		self.maxDistance = maxDistance

		self.bottom = bottom
		self.n_init = n_init
		self.status = OrderedDict()
		self.hits = OrderedDict()
		self.deregistered_obj = []

	def register(self, centroid):
		# when registering an object we use the next available object
		# ID to store the centroid
		self.objects[self.nextObjectID] = centroid
		self.disappeared[self.nextObjectID] = 0
		self.hits[self.nextObjectID] = 1

		self.status[self.nextObjectID] = "Tentative"
		if self.hits[self.nextObjectID] >= self.n_init:
			self.status[self.nextObjectID] = "Confirmed"
		self.nextObjectID += 1
	
	def deregister(self, objectID):
		# to deregister an object ID we delete the object ID from
		# both of our respective dictionaries
		del self.objects[objectID]
		del self.disappeared[objectID]
		del self.hits[objectID]

		if self.status[objectID] == "Confirmed":
			self.deregistered_obj.append(objectID)

		del self.status[objectID]
		
	def update(self, rects):
		# check to see if the list of input bounding box rectangles
		# is empty
		if rects is None or len(rects) == 0:
			# loop over any existing tracked objects and mark them
			# as disappeared
			for objectID in list(self.disappeared.keys()):
				self.disappeared[objectID] += 1

				# if we have reached a maximum number of consecutive
				# frames where a given object has been marked as
				# missing, deregister it
				if self.status[objectID] == "Confirmed" and self.disappeared[objectID] > self.maxDisappeared:
					self.deregister(objectID)
				elif self.status[objectID] == "Tentative" and self.disappeared[objectID] > self.maxDisappeared / 10:
					self.deregister(objectID)

			# return early as there are no centroids or tracking info
			# to update
			return self.objects
		
		rects = rects[:, :4] if isinstance(rects, np.ndarray) else rects.cpu().numpy()[:, :4]
		# initialize an array of input centroids for the current frame
		inputCentroids = np.zeros((len(rects), 4), dtype="int")

		# loop over the bounding box rectangles
		for (i, (startX, startY, endX, endY)) in enumerate(rects):
			# use the bounding box coordinates to derive the centroid
			cX = int((startX + endX) / 2.0)
			cY = int(endY) if self.bottom else int((startY + endY) / 2.0)
			w = int(endX - startX)
			h = int(endY - startY)
			inputCentroids[i] = (cX, cY, w, h)

		# if we are currently not tracking any objects take the input
		# centroids and register each of them
		if len(self.objects) == 0:
			for i in range(0, len(inputCentroids)):
				self.register(inputCentroids[i])

		# otherwise, are are currently tracking objects so we need to
		# try to match the input centroids to existing object
		# centroids
		else:
			# grab the set of object IDs and corresponding centroids
			objectIDs = list(self.objects.keys())
			objectCentroids = list(self.objects.values())

			# compute the distance between each pair of object
			# centroids and input centroids, respectively -- our
			# goal will be to match an input centroid to an existing
			# object centroid
			D = dist.cdist(np.array(objectCentroids)[:, :2], inputCentroids[:, :2])
			# print()
			# print('D: ')
			# print(D)
			# in order to perform this matching we must (1) find the
			# smallest value in each row and then (2) sort the row
			# indexes based on their minimum values so that the row
			# with the smallest value as at the *front* of the index
			# list
			rows = D.min(axis=1).argsort()
			# print('rows')
			# print(D.min(axis = 1))
			# print(rows)
			# next, we perform a similar process on the columns by
			# finding the smallest value in each column and then
			# sorting using the previously computed row index list
			cols = D.argmin(axis=1)[rows]
			# print('cols')
			# print(D.argmin(axis=1))
			# print(cols)
			# print()
			# in order to determine if we need to update, register,
			# or deregister an object we need to keep track of which
			# of the rows and column indexes we have already examined
			usedRows = set()
			usedCols = set()

			# loop over the combination of the (row, column) index
			# tuples
			for (row, col) in zip(rows, cols):
				# if we have already examined either the row or
				# column value before, ignore it
				if row in usedRows or col in usedCols:
					continue

				# if the distance between centroids is greater than
				# the maximum distance, do not associate the two
				# centroids to the same object
				if D[row, col] > self.maxDistance:
					continue

				# otherwise, grab the object ID for the current row,
				# set its new centroid, and reset the disappeared
				# counter
				objectID = objectIDs[row]
				self.objects[objectID] = inputCentroids[col]
				self.hits[objectID] += 1
				self.disappeared[objectID] = 0

				if self.status[objectID] == "Tentative" and self.hits[objectID] >= self.n_init:
					self.status[objectID] = "Confirmed"

				# indicate that we have examined each of the row and
				# column indexes, respectively
				usedRows.add(row)
				usedCols.add(col)

			# compute both the row and column index we have NOT yet
			# examined
			unusedRows = set(range(0, D.shape[0])).difference(usedRows)
			unusedCols = set(range(0, D.shape[1])).difference(usedCols)

			# in the event that the number of object centroids is
			# equal or greater than the number of input centroids
			# we need to check and see if some of these objects have
			# potentially disappeared
			if D.shape[0] >= D.shape[1]:
				# loop over the unused row indexes
				for row in unusedRows:
					# grab the object ID for the corresponding row
					# index and increment the disappeared counter
					objectID = objectIDs[row]
					self.disappeared[objectID] += 1

					# check to see if the number of consecutive
					# frames the object has been marked "disappeared"
					# for warrants deregistering the object
					if self.status[objectID] == "Confirmed" and self.disappeared[objectID] > self.maxDisappeared:
						self.deregister(objectID)
					elif self.status[objectID] == "Tentative" and self.disappeared[objectID] > self.maxDisappeared / 10:
						self.deregister(objectID)

			# otherwise, if the number of input centroids is greater
			# than the number of existing object centroids we need to
			# register each new input centroid as a trackable object
			else:
				for col in unusedCols:
					self.register(inputCentroids[col])

		# return the set of trackable objects
		return self.objects