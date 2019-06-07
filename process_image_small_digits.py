import numpy as np
import cv2
import imutils
from ImageProcessing import FrameProcessorSmall, ProcessingVariables
from DisplayUtils.TileDisplay import show_img, reset_tiles
import os
import random

std_height = 300

# thresh = 73  # 1-50 mod 2 25
# erode = 3  # 3-4 2
# adjust = 15  # 10-40 30
# blur = 9  # 5-15 mod 2 7

erode = 4
threshold = ProcessingVariables.threshold
adjustment = ProcessingVariables.adjustment
iterations = ProcessingVariables.iterations
blur = 7

version = '_2_8'
test_folder = ''

frameProcessor = FrameProcessorSmall(std_height, version, False, write_digits=False)
def read_image(img):
    break_fully = False
    output = ''
    debug_images = []
    #frameProcessor = FrameProcessor(std_height, version, False)
    frameProcessor.set_image(img)
    frameProcessor.set_file_name(str(int(random.random()*10000)))
    for blur in [25,23,21,19,17,15,13,11,9,7,5,3,1]:
	if break_fully:
		break
    	for erode in [1,2,3,4,5,6,7]:
		if break_fully:
			break
		for iterations in [4,3,2,1]:
			try:	
				output = frameProcessor.process_image(blur, threshold, adjustment, erode, iterations)
				if output != '' and ((len(output) == 2 and '.' not in output) or (len(output) == 3 and '.' in output) or (len(output) == 4 and '.' in output) or (len(output) == 3 and '.' not in output) ) and (check_instance(output, float) or check_instance(output, int)):
					output=int(output.replace(".",""))*1.0/10
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
	output = ''
	for val in [17,37,57]:
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
	reduce_height = height*33/100
	reduce_width = width*50/100
	img_glucose = img2[height-reduce_height:height, 0:reduce_width] 
	#cv2.imwrite(str(i)+".jpg", img_glucose)		
	output = read_image(img2)
	return output
							

id=0
correct=0
incorrect=0
for f in os.listdir("/home/kiril/Downloads/SDB Device Output Images (2)/Camera with Flash Iphone 8 small/"):
	id = id+1
	img = cv2.imread("/home/kiril/Downloads/SDB Device Output Images (2)/Camera with Flash Iphone 8 small/"+f)
	if str(f.replace("_",".").replace("*","").replace(".JPG", "").replace(".jpg", "")) == str(ocr_image(img, "cam"+str(id))):
              correct = correct+1
	      print "Success: " + str(correct)
	else:
	      print f
	      print ocr_image(img, "flu"+str(id))
	      incorrect = incorrect +1

for f in os.listdir("/home/kiril/Downloads/SDB Device Output Images (2)/Fluorescent Indoor small/"):
	id = id+1
	img = cv2.imread("/home/kiril/Downloads/SDB Device Output Images (2)/Fluorescent Indoor small/"+f)
	if str(f.replace("_",".").replace("*","").replace(".JPG", "").replace(".jpg", "")) == str(ocr_image(img, "flu"+str(id))):
              correct = correct+1
	      print "Success: " + str(correct)
	else:
	      print f
	      print ocr_image(img, "flu"+str(id))
	      incorrect = incorrect +1

for f in os.listdir("/home/kiril/Downloads/SDB Device Output Images (2)/Natural Light small/"):
	id = id+1
	img = cv2.imread("/home/kiril/Downloads/SDB Device Output Images (2)/Natural Light small/"+f)
	if str(f.replace("_",".").replace("*","").replace(".JPG", "").replace(".jpg", "")) == str(ocr_image(img, "nat"+str(id))):
              correct = correct+1
	      print "Success: " + str(correct)
	else:
	      print f
	      print ocr_image(img, "flu"+str(id))
	      incorrect = incorrect +1

accuracy = correct*1.0/id
print "Accuracy:"+str(accuracy)

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




