import numpy as np
import cv2
import imutils
from skimage import exposure
from pytesseract import image_to_string
import PIL

def take_picture(should_save=False, d_id=0):
  cam = cv2.VideoCapture(d_id)
  s, img = cam.read()
  if s:
    if should_save:
      cv2.imwrite('ocr.jpg',img)
    print "picture taken"
  return img

def cnvt_edged_image(img_arr, should_save=False):
  # ratio = img_arr.shape[0] / 300.0
  image = imutils.resize(img_arr,height=300)
  gray_image = cv2.bilateralFilter(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),11, 17, 17)
  edged_image = cv2.Canny(gray_image, 30, 200)

  if should_save:
    cv2.imwrite('cntr_ocr.jpg')

  return edged_image

'''image passed in must be ran through the cnv_edge_image first'''
def find_display_contour(edge_img_arr):
  display_contour = None
  edge_copy = edge_img_arr.copy()
  _im2, contours, hierarchy = cv2.findContours(edge_copy, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  top_cntrs = sorted(contours, key = cv2.contourArea, reverse = True)[:10]

  for cntr in top_cntrs:
    peri = cv2.arcLength(cntr,True)
    approx = cv2.approxPolyDP(cntr, 0.02 * peri, True)

    if len(approx) == 4:
      display_contour = approx
      break

  return display_contour

def crop_display(image_arr):
  edge_image = cnvt_edged_image(image_arr)
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

def process_image(orig_image_arr):
  ratio = orig_image_arr.shape[0] / 300.0

  display_image_arr = normalize_contrs(orig_image_arr,crop_display(orig_image_arr))
  #display image is now segmented.
  #gry_disp_arr = cv2.cvtColor(display_image_arr, cv2.COLOR_BGR2GRAY)
  #cv2.imshow("image1.jpg", gry_disp_arr)
  #cv2.waitKey(0)
  #gry_disp_arr = exposure.rescale_intensity(gry_disp_arr, out_range= (0,255))

  #thresholding
  #ret, thresh = cv2.threshold(gry_disp_arr,127,255,cv2.THRESH_BINARY)
  return display_image_arr

def ocr_image(orig_image_arr,i):
  try:
  	img = process_image(orig_image_arr)
  except:
	for degree in [1.5, 1.8, 2.5, 2.8, 3.5, 3.8, 4.5, 4.8, 5.5, 5.8, 6.5]:
		try:
			orig_image_arr2 = cv2.multiply(orig_image_arr, np.array([degree]))
			img = process_image(orig_image_arr2)
			break
		except:
			img = orig_image_arr
  kernel = np.ones((3, 3), np.uint8)
  img2 = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
  img = cv2.resize(img,None,fx=0.3,fy=0.3)
  cv2.imwrite("image"+i+".jpg", img)
  otsu_thresh_image = PIL.Image.fromarray(img)

for i in range(59):
	id = i+60
	img = cv2.imread("/home/kiril/Downloads/SDB Device Output Images/all_examples2/example"+str(id)+".jpg")
	print id
	ocr_image(img, str(id))
