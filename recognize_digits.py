import cv2
from ImageProcessing import FrameProcessor, ProcessingVariables
from DisplayUtils.TileDisplay import show_img, reset_tiles

def process_image(img):
    break_fully = False
    output = ''
    debug_images = []
    for blur in [1,2,3,4,5,6,7,8,9]:
	print(break_fully)
	if break_fully:
		break
    	for erode in [1,2,3,4,5,6,7]:
		if break_fully:
			break
		for iterations in [1,2,3,4]:
			print "Blur:"+str(blur)
			print "Erode:"+str(erode)
			print "Iterations:"+str(iterations)
			try:	
				debug_images, output = frameProcessor.process_image(blur, threshold, adjustment, erode, iterations)
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
