import cv2
import numpy as np
import subprocess
import os

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
	Helper function to print out images, for debugging. Pass them in as a list.
	Press any key to continue.
	"""
	cv2.imshow("image", img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def cd_color_segmentation(img, template):
	"""
	Implement the cone detection using color segmentation algorithm
	Input:
		img: np.3darray; the input image with a cone to be detected. BGR.
		template_file_path; Not required, but can optionally be used to automate setting hue filter values.
	Return:
		bbox: ((x1, y1), (x2, y2)); the bounding box of the cone, unit in px
				(x1, y1) is the top left of the bbox and (x2, y2) is the bottom right of the bbox
	"""
	########## YOUR CODE STARTS HERE ##########
	# orange_hue_range = (15,39)
	# orange_sat_range = (75,100)
	# orange_value_range = (80,100)
	# 7,8,9: (0,195, 200) -> (30,300,300) -----> Works for the rest too
# Best test score(so far): [0,163,136] --> (30,300,300)
	orange_lower = np.array([0,225,140])
	orange_upper = np.array([30,300,300])
	bounding_box = ((0,0),(0,0))
	# image_in = cv2.imread(img)

	image_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	# image_print(image_hsv)

	h, s, v = cv2.split(image_hsv)
	clahe = cv2.createCLAHE(clipLimit=2.03, tileGridSize=(8, 8))
	v_eq = clahe.apply(v)
	hsv_eq = cv2.merge([h, s, v_eq])
	# image_print(hsv_eq)
	# image_print(image_hsv)
	# print(image_hsv[-1])
	image_orange = cv2.inRange(hsv_eq, orange_lower, orange_upper)
	# print(image_orange)
	# image_print(image_orange)
	kernel = np.ones((3,3), np.uint8)
	# image_orange = cv2.erode(image_orange, kernel, iterations=2)
	# image_orange = cv2.dilate(image_orange, kernel, iterations=2)
	kernel = np.ones((3,3), np.uint8)
	# image_orange = cv2.erode(image_orange, kernel, iterations=1)
	for _ in range(20):
		# image_orange = cv2.erode(image_orange, kernel, iterations=10)
		image_orange = cv2.morphologyEx(image_orange, cv2.MORPH_OPEN, kernel)
		image_orange = cv2.morphologyEx(image_orange, cv2.MORPH_CLOSE, kernel)
	# image_orange = cv2.erode(image_orange, kernel, iterations=3)
	image_orange = cv2.dilate(image_orange, kernel, iterations=11)
	# image_orange = cv2.erode(image_orange, kernel, iterations=10)
		# image_orange = cv2.morphologyEx(image_orange, cv2.MORPH_OPEN, kernel)
		# image_orange = cv2.morphologyEx(image_orange, cv2.MORPH_CLOSE, kernel)
	# mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)
	# image_print(image_orange)
	# edged = cv2.Canny(image_orange, 0, 300)
	# image_print(image_orange)
	# ---- Watershed Segmentation ----
    # 1. Determine sure background by dilating the mask.
	# sure_bg = cv2.dilate(image_orange, kernel, iterations=3)

    # 2. Compute the distance transform to find sure foreground.
	dist_transform = cv2.distanceTransform(image_orange, cv2.DIST_L2, 5)
	ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
	sure_fg = np.uint8(sure_fg)

    # 3. Identify unknown regions (i.e., where background and foreground overlap).
	unknown = cv2.subtract(image_orange, sure_fg)

    # 4. Marker labelling: connected components on sure foreground.
	ret, markers = cv2.connectedComponents(sure_fg)
	markers = markers + 1  # Increment markers so that the background is not 0.
	markers[unknown == 255] = 0

    # 5. Apply the watershed algorithm.
	markers = cv2.watershed(img, markers)
    # Mark boundaries on the original image (boundaries are marked with -1).
	img[markers == -1] = [255, 0, 0]  # Red boundaries for visualization.

    # Create a mask from the watershed markers: use regions with marker value >1.
	segmented_mask = np.zeros_like(image_orange)
	segmented_mask[markers > 1] = 255
	# image_print(segmented_mask)
    # ---- End Watershed Segmentation ----


	contours, _ = cv2.findContours(segmented_mask,  
    cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
	largest_contour = max(contours, key=cv2.contourArea)
	# print(contours[0].shape)
	cv2.drawContours(img, contours, -1, (0, 255, 0), 3) 
	# image_print(segmented_mask)
	# image_print(img)
	# print(len(contours))
	x, y, w, h = cv2.boundingRect(largest_contour)
	center_x = x + w / 2
	center_y = y + h / 2

	# Scale factor for 5% bigger.
	scale = 1.05

	# Compute the new dimensions.
	w_new = int(w * scale)
	h_new = int(h * scale)

	# Compute the new top-left coordinates so that the rectangle remains centered.
	x_new = int(center_x - w_new / 2)
	y_new = int(center_y - h_new / 2)

	# Ensure the new rectangle stays within the image boundaries.
	x_new = max(x_new, 0)
	y_new = max(y_new, 0)
	if x_new + w_new > img.shape[1]:
		w_new = img.shape[1] - x_new
	if y_new + h_new > img.shape[0]:
		h_new = img.shape[0] - y_new

	# Draw the expanded bounding rectangle.
	cv2.rectangle(img, (x_new, y_new), (x_new + w_new, y_new + h_new), 128, 2)

	image_print(img)
	bounding_box = ((x_new, y_new), (x_new + w_new, y_new + h_new))
	

	########### YOUR CODE ENDS HERE ###########

	# Return bounding box
	return bounding_box
test_im = cv2.imread('/Users/paul/racecar_docker/home/racecar_ws/src/visual_servoing/visual_servoing/visual_servoing/computer_vision/test_images_cone/test9.jpg')
cd_color_segmentation(test_im, None)



def optimize():
	# print("Running from:", os.getcwd())
	try:
		result = subprocess.run(
			['python3', '-u', 'cv_test.py', 'cone', 'color'],
			capture_output=True,
			text=True,
			timeout=100  # seconds
		)
		# print("Stdout:", result.stdout)
		# print(type(result.stdout))
		out = result.stdout.split('\n')[-21:-1]
		tot = 0
		for test in out:
			score = float(test.split(".jpg', ")[1].split(')')[0])
			tot += score
		return tot/20
			# print(score)

		# print(len(out))
		# print(out)
		# print("Stderr:", result.stderr)
	except:
		subprocess.TimeoutExpired
		print("Subprocess timed out.")

if __name__ == '__main__':
	# print(optimize())
	pass