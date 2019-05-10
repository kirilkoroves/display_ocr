import os
import time

from DisplayUtils.Colors import bcolors
from DisplayUtils.TileDisplay import show_img, reset_tiles
from ImageProcessing import FrameProcessor, ProcessingVariables
import digital_display_ocr
import cv2
std_height = 90

# thresh = 73  # 1-50 mod 2 25
# erode = 3  # 3-4 2
# adjust = 15  # 10-40 30
# blur = 9  # 5-15 mod 2 7

erode = ProcessingVariables.erode
threshold = ProcessingVariables.threshold
adjustment = ProcessingVariables.adjustment
iterations = ProcessingVariables.iterations
blur = ProcessingVariables.blur

version = '_2_0'

frameProcessor = FrameProcessor(std_height, version, False, write_digits=False)
orig_image_arr = cv2.imread("20190315_204854.jpg")
image = digital_display_ocr.process_image(orig_image_arr)
debug_images, output = frameProcessor.process_image(image, blur, threshold, adjustment, erode, iterations)
for image in debug_images:
  show_img(image[0], image[1])
print output
cv2.waitKey()
