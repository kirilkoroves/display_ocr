import numpy as np
import cv2
from ImageProcessing import FrameProcessor, ProcessingVariables
from DisplayUtils.TileDisplay import show_img, reset_tiles

std_height = 400
version = "_2_2"
threshold = ProcessingVariables.threshold
adjustment = ProcessingVariables.adjustment
iterations = ProcessingVariables.iterations

def read_image(img):
    break_fully = False
    output = ''
    debug_images = []
    frameProcessor = FrameProcessor(std_height, version, False)
    frameProcessor.set_image(img)
    for blur in [1,3,5,7,9]:
		if break_fully:
			break
		for erode in [1,2,3,4,5,6,7]:
			if break_fully:
				break
			for iterations in [1,2,3,4]:
				try:
					output = frameProcessor.process_image(blur, threshold, adjustment, erode, iterations)
					if '.' in output and len(output) == 4:
						output = remove_duplicate_chars(output)
					elif '.' not in output and len(output) == 3:
						output = remove_duplicate_chars(output)
					if output != '' and ((len(output) == 2 and '.' not in output) or (len(output) == 3 and '.' in output) ) and (check_instance(output, float) or check_instance(output, int)):
						break_fully = True
						break
				except:
					output = output
    return output

def remove_duplicate_chars(output):
	s = []	
	for i in range(0, len(output)):
		char = output[i]
		if char not in s and char not in output[i+1:i+2]:
			s.append(char)
	return "".join(s)


def check_instance(val, val_type):
	try:	
		val_type(val)
		return True
	except:
		return False

def cnvt_edged_image(img_arr, val, should_save=False,):
	# ratio = img_arr.shape[0] / 300.0
	image = imutils.resize(img_arr,height=300)
	gray_image = cv2.bilateralFilter(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),11, val, val)
	edged_image = cv2.Canny(gray_image, 20, 200)
	return edged_image

'''image passed in must be ran through the cnv_edge_image first'''
def find_display_contour(edge_img_arr):
	display_contour = None
	edge_copy = edge_img_arr.copy()
	_im2, contours, hierarchy = cv2.findContours(edge_copy, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	top_cntrs = sorted(contours, key = cv2.contourArea, reverse = True)[:10]

	for cntr in top_cntrs:
		peri = cv2.arcLength(cntr,True)
		approx = cv2.approxPolyDP(cntr, 0.01 * peri, True)

		if len(approx) == 4:
	 		display_contour = approx
	  		break

	return display_contour

def crop_display(image_arr, val):
  edge_image = cnvt_edged_image(image_arr, val)
  display_contour = find_display_contour(edge_image)
  cntr_pts = display_contour.reshape(4,2)
  return cntr_pts


def normalize_contrs(img,cntr_pts):
	ratio = img.shape[0] / 300.0
	norm_pts = np.zeros((4,2), dtype="float32")

	s = cntr_pts.sum(axis=1)
	norm_pts[0] = cntr_pts[np.argmin(s)]
	norm_pts[2] = cntr_pts[np.argmax(s)]

	d = np.diff(cntr_pts,axis=1)
	norm_pts[1] = cntr_pts[np.argmin(d)]
	norm_pts[3] = cntr_pts[np.argmax(d)]

	norm_pts *= ratio

	(top_left, top_right, bottom_right, bottom_left) = norm_pts

	width1 = np.sqrt(((bottom_right[0] - bottom_left[0]) ** 2) + ((bottom_right[1] - bottom_left[1]) ** 2))
	width2 = np.sqrt(((top_right[0] - top_left[0]) ** 2) + ((top_right[1] - top_left[1]) ** 2))
	height1 = np.sqrt(((top_right[0] - bottom_right[0]) ** 2) + ((top_right[1] - bottom_right[1]) ** 2))
	height2 = np.sqrt(((top_left[0] - bottom_left[0]) ** 2) + ((top_left[1] - bottom_left[1]) ** 2))

	max_width = max(int(width1), int(width2))
	max_height = max(int(height1), int(height2))

	dst = np.array([[0,0], [max_width -1, 0],[max_width -1, max_height -1],[0, max_height-1]], dtype="float32")
	persp_matrix = cv2.getPerspectiveTransform(norm_pts,dst)
	return cv2.warpPerspective(img,persp_matrix,(max_width,max_height))

def process_image(orig_image_arr, val):
	ratio = orig_image_arr.shape[0] / 300.0
	display_image_arr = normalize_contrs(orig_image_arr,crop_display(orig_image_arr, val))
	return display_image_arr

def ocr_image(orig_image_arr,i):
	output_g6pd = ''
	output_glucose = ''
	found_g6pd = False
	found_glucose = False
	for val in [17,37,57]:
		if found_g6pd == True or found_glucose == True:
			break
		try:
			img = process_image(orig_image_arr, val)
		except:
			for degree in [1.5, 1.8, 2.5, 2.8, 3.5, 3.8, 4.5, 4.8, 5.5, 5.8, 6.5]:
				try:
					orig_image_arr2 = cv2.multiply(orig_image_arr, np.array([degree]))
					img = process_image(orig_image_arr2, val)
					break
				except:
					img = orig_image_arr
		
		kernel = np.ones((3, 3), np.uint8)
		img2 = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
		height, width, channels = img2.shape
		if found_g6pd == False:
			reduce_height = height*15/100
			reduce_width = width*50/100
			img_g6pd = img2[reduce_height:height-reduce_height, reduce_width:width]
			img_g6pd = cv2.resize(img_g6pd,None,fx=0.7,fy=0.7)
			output_g6pd = read_image(img_g6pd)
			if output_g6pd != '' and ((len(output_g6pd) == 2 and '.' not in output_g6pd) or (len(output_g6pd) == 3 and '.' in output_g6pd) ) and (check_instance(output_g6pd, float) or check_instance(output_g6pd, int)):
				if '.' not in output_g6pd and len(output_g6pd) == 2 and check_instance(output_g6pd, int):
					output_g6pd = int(output_g6pd) * 1.0 / 10
					found_g6pd = True
				elif len(output_g6pd) == 3 and output_g6pd[0] == '.' and check_instance(output_g6pd, float):
					output_g6pd = float(output_g6pd) * 100 * 1.0 / 10
					found_g6pd = True

		if found_glucose == False:
			reduce_height = height*15/100
			reduce_width = width*50/100
			img_glucose = img2[height-reduce_height:height, 0:reduce_width]
			output_glucose = read_image(img_glucose)
			if output_glucose != '' and (len(output_glucose) == 2 or len(output_glucose) == 3 or len(output_glucose) == 4 ) and (check_instance(output_glucose, float) or check_instance(output_glucose, int)):
				if '.' not in output_glucose and len(output_glucose) == 2 and check_instance(output_glucose, int):
					output_glucose = int(output_glucose) * 1.0 / 10
				elif len(output_glucose) == 3 and output_glucose[0] == '.' and check_instance(output_glucose, float):
					output_glucose = float(output_glucose) * 100 * 1.0 / 10
				elif len(output_glucose) == 4 and output_glucose[0] == '.' and check_instance(output_glucose, float):
					output_glucose = float(output_glucose) * 1000 * 1.0 / 10
					found_glucose = True

	return output_g6pd, output_glucose
							


for i in range(5):
	id = i+1
	img = cv2.imread("/home/kiril/Downloads/SDB Device Output Images/all_examples/example"+str(id)+".jpg")
	print id
	output_g6pd, output_glucose = ocr_image(img, str(id))
	print output_g6pd
	print output_glucose

#for i in range(60):
#  id = i+60
#  img = cv2.imread("/home/kiril/Downloads/SDB Device Output Images/all_examples2/example"+str(id)+".jpg")
#  print id
#  print ocr_image(img, str(id))

#for i in range(37):
#	id = i+120
#	img = cv2.imread("/home/kiril/Downloads/SDB Device Output Images/all_examples3/example"+str(id)+".jpg")
#	print id
#	print ocr_image(img, str(id))

#for i in range(47):
#	id = i+158
#	img = cv2.imread("/home/kiril/Downloads/SDB Device Output Images/all_examples4/example"+str(id)+".jpg")
#	print id
#	print ocr_image(img, str(id))


