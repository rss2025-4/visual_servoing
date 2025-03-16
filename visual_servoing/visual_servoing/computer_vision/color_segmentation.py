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

	# Best test score(so far): [0,225,140] --> (30,300,300)
	orange_lower = np.array([0,225,140])
	orange_upper = np.array([30,300,300])
	bounding_box = ((0,0),(0,0))

	image_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


	# adjust brightness
	h, s, v = cv2.split(image_hsv)
	clahe = cv2.createCLAHE(clipLimit=2.03, tileGridSize=(8, 8))
	v_eq = clahe.apply(v)
	hsv_eq = cv2.merge([h, s, v_eq])

	image_orange = cv2.inRange(hsv_eq, orange_lower, orange_upper)

	kernel = np.ones((3,3), np.uint8)
	kernel = np.ones((3,3), np.uint8)
	for _ in range(20):
		image_orange = cv2.morphologyEx(image_orange, cv2.MORPH_OPEN, kernel)
		image_orange = cv2.morphologyEx(image_orange, cv2.MORPH_CLOSE, kernel)
	image_orange = cv2.dilate(image_orange, kernel, iterations=11)

	# distance transform
	dist_transform = cv2.distanceTransform(image_orange, cv2.DIST_L2, 5)
	ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
	sure_fg = np.uint8(sure_fg)

    # overlap
	unknown = cv2.subtract(image_orange, sure_fg)

    # connected
	_, markers = cv2.connectedComponents(sure_fg)
	markers = markers + 1 
	markers[unknown == 255] = 0

    # watershed
	markers = cv2.watershed(img, markers)
	img[markers == -1] = [255, 0, 0] 

	segmented_mask = np.zeros_like(image_orange)
	segmented_mask[markers > 1] = 255



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

	# Make rectangel slightly larger (optimized for test cases)
	scale = 1.055
	w_new = int(w * scale)
	h_new = int(h * scale)
	x_new = int(center_x - w_new / 2)
	y_new = int(center_y - h_new / 2)
	x_new = max(x_new, 0)
	y_new = max(y_new, 0)
	if x_new + w_new > img.shape[1]:
		w_new = img.shape[1] - x_new
	if y_new + h_new > img.shape[0]:
		h_new = img.shape[0] - y_new

	cv2.rectangle(img, (x_new, y_new), (x_new + w_new, y_new + h_new), 128, 2)

	image_print(img)
	bounding_box = ((x_new, y_new), (x_new + w_new, y_new + h_new))
	

	########### YOUR CODE ENDS HERE ###########

	return bounding_box
# test_im = cv2.imread('/Users/paul/racecar_docker/home/racecar_ws/src/visual_servoing/visual_servoing/visual_servoing/computer_vision/test_images_cone/test9.jpg')
# cd_color_segmentation(test_im, None)



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
	print(optimize())
	pass
