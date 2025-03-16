import cv2
import imutils
import numpy as np
import matplotlib.pyplot as plt


#################### X-Y CONVENTIONS #########################
# 0,0  X  > > > > >
#
#  Y
#
#  v  This is the image. Y increases downwards, X increases rightwards
#  v  Please return bounding boxes as ((xmin, ymin), (xmax, ymax))
#  v
#  v
#  v
###############################################################

def image_print(img):
	"""
	Helper function to print out images, for debugging.
	Press any key to continue.
	"""
	winname = "Image"
	cv2.namedWindow(winname)        # Create a named window
	cv2.moveWindow(winname, 40,30)  # Move it to (40,30)
	cv2.imshow(winname, img)
	cv2.waitKey()
	cv2.destroyAllWindows()

def cd_sift_ransac(img, template):
	"""
	Implement the cone detection using SIFT + RANSAC algorithm
	Input:
		img: np.3darray; the input image with a cone to be detected
	Return:
		bbox: ((x1, y1), (x2, y2)); the bounding box of the cone, unit in px
				(x1, y1) is the bottom left of the bbox and (x2, y2) is the top right of the bbox
	"""
	# Minimum number of matching features
	MIN_MATCH = 10 # Adjust this value as needed
	# Create SIFT
	sift = cv2.xfeatures2d.SIFT_create()

	# Compute SIFT on template and test image
	# kp is the keypoints and des is the feature descriptor
	kp1, des1 = sift.detectAndCompute(template,None)
	kp2, des2 = sift.detectAndCompute(img,None)

	# Find matches
	bf = cv2.BFMatcher()
	matches = bf.knnMatch(des1,des2,k=2)

	# Find and store good matches
	good = []
	for m,n in matches:
		if m.distance < 0.75*n.distance:
			good.append(m)

	# If enough good matches, find bounding box
	if len(good) > MIN_MATCH:
		src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
		dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

		# Create mask
		M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
		matchesMask = mask.ravel().tolist()
		# height and width of template
		h, w = template.shape
		# corresponds to top left, bottom left, bottom right, top right
		pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)

		########## YOUR CODE STARTS HERE ##########

		bounding_box = cv2.perspectiveTransform(pts, M)
		print(f"[SIFT] matches: ", len(good), "bounding box: ", bounding_box)
		# Extract x and y coordinates from the transformed corners
		x = bounding_box[:,0,0]
		y = bounding_box[:,0,1]
		print(f"[SIFT] x: ", x, "y: ", y)

		x_min, x_max = int(np.min(x)), int(np.max(x))
		y_min, y_max = int(np.min(y)), int(np.max(y))
		########### YOUR CODE ENDS HERE ###########

		# Return bounding box
		return ((x_min, y_min), (x_max, y_max))
	else:

		print(f"[SIFT] not enough matches; matches: ", len(good))

		# Return bounding box of area 0 if no match found
		return ((0,0), (0,0))

def cd_template_matching(img, template, img_path, template_path):
	"""
	Implement the cone detection using template matching algorithm
	Input:
		img: np.3darray; the input image with a cone to be detected
	Return:
		bbox: ((x1, y1), (x2, y2)); the bounding box of the cone, unit in px
				(x1, y1) is the bottom left of the bbox and (x2, y2) is the top right of the bbox
	"""
	# this does canny edge detection on the template to emphasize 
	# edges of the cone in the template so that it can be matched
	# with cone like objects in the test image
	template_canny = cv2.Canny(template, 50, 200)

	# Perform Canny Edge detection on test image
	grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	img_canny = cv2.Canny(grey_img, 50, 200)

	# Get dimensions of template ? (or is it the test image)
	(img_height, img_width) = img_canny.shape[:2]
	(tH, tW) = template_canny.shape[:2]

	# Keep track of best-fit match
	best_match = None
	img_path = img_path.split("/")[-1]
	template_path = template_path.split("/")[-1]

	# Loop over different scales of the template
	for scale in np.linspace(1.5, .5, 50):
		# Resize the image
		resized_template = imutils.resize(template_canny, width = int(template_canny.shape[1] * scale))
		(h,w) = resized_template.shape[:2]
		# Check to see if test image is now smaller than template image
		if resized_template.shape[0] > img_height or resized_template.shape[1] > img_width:
			continue

		########## YOUR CODE STARTS HERE ##########
		# Use OpenCV template matching functions to find the best match
		# across template scales.

		# Remember to resize the bounding box using the highest scoring scale
		# x1,y1 pixel will be accurate, but x2,y2 needs to be correctly scaled
		resized_canny = cv2.Canny(resized_template, 50, 200)
		# sqdiff sucks 
		#methods = ['TM_CCOEFF', 'TM_CCOEFF_NORMED', 'TM_CCORR', 'TM_CCORR_NORMED', 'TM_SQDIFF', 'TM_SQDIFF_NORMED']
		methods = ['TM_CCOEFF_NORMED']
		for m in methods:
			method = getattr(cv2, m)
			res = cv2.matchTemplate(img_canny, resized_canny, method)
			(_, max_val, _, max_loc) = cv2.minMaxLoc(res)
			#  DEBUG
			# visualized_match = img.copy()
			# cv2.rectangle(visualized_match, (max_loc[0], max_loc[1]),
			# 	(max_loc[0] + int(tW *scale), max_loc[1] + int(tH*scale)), (255, 0, 0), 3)
			# 	# Show with Matplotlib
			# plt.figure(figsize=(8, 6))
			# plt.imshow(cv2.cvtColor(visualized_match, cv2.COLOR_BGR2RGB))
			# plt.title("Template Matching Result")
			# plt.axis("off")
			# plt.show()
			# save_path = f"template_match_outputs/BEST_img:{img_path}_{m}_template{template_path}_{scale}x.png"
			# plt.savefig(save_path)
			# print(f"Plot saved to {save_path}")
			# plt.close()
			if best_match is None or max_val > best_match[0]:
				print(f"found a better match: ", max_val)
				best_match = (max_val, max_loc, scale)
				max_val, max_loc, best_scale = best_match
				# clone = np.dstack([resized_canny, resized_canny, resized_canny])
				
				
	print(f"[Template] best match: ", best_match)
	if best_match:
		max_val, max_loc, best_scale = best_match
		
		x1, y1 = max_loc
		w, h = int(tW*best_scale), int(tH*best_scale)
		x2, y2 = (x1 + w, y1 + h)
		bounding_box = ((x1, y1), (x2, y2))
		visualized_match = img.copy()
		cv2.rectangle(visualized_match, (max_loc[0], max_loc[1]),
			(max_loc[0] + int(tW *scale), max_loc[1] + int(tH*scale)), (255, 0, 0), 3)
			# Show with Matplotlib
		plt.figure(figsize=(8, 6))
		plt.imshow(cv2.cvtColor(visualized_match, cv2.COLOR_BGR2RGB))
		plt.title("Template Matching Result")
		plt.axis("off")
		plt.show()
		save_path = f"template_match_outputs/BEST_img:{img_path}_{m}_template{template_path}_{scale}x.png"
		plt.savefig(save_path)
		print(f"Plot saved to {save_path}")
		plt.close()
	else:
		print(f"[Template] no match found")
		bounding_box = ((0,0),(0,0))
		########### YOUR CODE ENDS HERE ###########
	print(f"[Template] bounding box: ", bounding_box)
	return bounding_box
