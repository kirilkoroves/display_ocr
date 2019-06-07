import cv2
import numpy as np
import os
from OpenCVUtils import inverse_colors, sort_contours
from warp.runtime import config

RESIZED_IMAGE_WIDTH = 20
RESIZED_IMAGE_HEIGHT = 30
CROP_DIR = 'crops'


def increase_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img

def average_brightness(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    (a, b, c, d) = cv2.mean(hsv[:, :, 2])
    return a

class FrameProcessorSmall:
    def __init__(self, height, version, debug=True, write_digits=True):
        self.debug = False
        self.version = version
        self.height = height
        self.file_name = None
        self.img = None
        self.width = 0
        self.original = None
        self.write_digits = True
        self.min_canny = 10
        self.max_canny = 20
        self.brightness = 0
        self.knn = self.train_knn(self.version)

    def set_image(self, img):
        self.img = img
        height, width, channels = self.img.shape
        self.original, self.width = self.resize_to_height(self.height)
        self.img = self.original.copy()

    def set_file_name(self, file_name):
        self.file_name = file_name
        height, width, channels = self.img.shape
        self.original, self.width = self.resize_to_height(self.height)
        self.img = self.original.copy()


    def resize_to_height(self, height):
        r = self.img.shape[0] / float(height)
        dim = (int(self.img.shape[1] / r), height)
        img = cv2.resize(self.img, dim, interpolation=cv2.INTER_AREA)
        return img, dim[0]

    def train_knn(self, version):
        npa_classifications = np.loadtxt("knn/classifications" + version + ".txt",
                                         np.float32)  # read in training classifications
        npa_flattened_images = np.loadtxt("knn/flattened_images" + version + ".txt",
                                          np.float32)  # read in training images

        npa_classifications = npa_classifications.reshape((npa_classifications.size, 1))
        k_nearest = cv2.ml.KNearest_create()
        k_nearest.train(npa_flattened_images, cv2.ml.ROW_SAMPLE, npa_classifications)
        return k_nearest

    def process_image(self, blur, threshold, adjustment, erode, iterations):

        self.img = self.original.copy()
        self.img = increase_brightness(self.img, self.brightness)
        debug_images = []
        img2gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
    	debug_images.append(('Gray',img2gray))
        # Blur to reduce noise
        img_blurred = cv2.GaussianBlur(img2gray, (blur, blur), 0)
        #debug_images.append(('Blurred', img_blurred))

        #cropped = img_blurred

        # Threshold the image
        cropped_threshold = cv2.adaptiveThreshold(img_blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
                                                  threshold, adjustment)
        debug_images.append(('Cropped Threshold', cropped_threshold))

        # Erode the lcd digits to make them continuous for easier contouring
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (erode, erode))
        eroded = cv2.erode(cropped_threshold, kernel, iterations=iterations)
        debug_images.append(('Eroded', eroded))
    
        #closing = cv2.morphologyEx(erode, cv2.MORPH_CLOSE, kernel)
        #debug_images.append(('Dilate', closing))
        # Reverse the image to so the white text is found when looking for the contours
        v = np.median(eroded)
 
        # apply automatic Canny edge detection using the computed median
        #sigma=0.33
        #lower = int(max(100, (1.0 - sigma) * v))
        #upper = int(min(200, (1.0 + sigma) * v))
        canny = cv2.Canny(eroded, self.min_canny, self.max_canny)

        debug_images.append(('Canny', canny))

        # Find the lcd digit contours
        _, contours, _ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # get contours

        # Assuming we find some, we'll sort them in order left -> right
        if len(contours) > 0:
            contours, _ = sort_contours(contours)

        potential_decimals = []
        potential_digits = []

        total_digit_height = 0
        total_digit_y = 0

        # Aspect ratio for all non 1 character digits
        desired_aspect = 0.6
        # Aspect ratio for the "1" digit
        digit_one_aspect = 0.35
        # The allowed buffer in the aspect when determining digits
        aspect_buffer = 0.15

        # Loop over all the contours collecting potential digits and decimals
        cv2.drawContours(self.img, contours, -1, (0,255,0), 3)        
        for contour in contours:
            # get rectangle bounding contour
            [x, y, w, h] = cv2.boundingRect(contour)

            aspect = float(w) / h
            size = w * h

            # It's a square, save the contour as a potential digit
            if size > 100 and aspect >= 1 - .3 and aspect <= 1 + .3:
                potential_decimals.append(contour)

            # If it's small and it's not a square, kick it out
            if size < 20 * 100 and (aspect < 1 - aspect_buffer and aspect > 1 + aspect_buffer):
                continue

            # Ignore any rectangles where the width is greater than the height
            if w > h:
                if self.debug:
                    cv2.rectangle(self.img, (x, y), (x + w, y + h), (0, 0, 255), 2)
                continue

            # If the contour is of decent size and fits the aspect ratios we want, we'll save it
	    #print("SIZE:"+str(size)+", X:"+str(x)+", Y:"+str(y))
            if ((size < 20000 and size > 5000 and aspect >= desired_aspect - aspect_buffer and aspect <= desired_aspect + aspect_buffer) or
                (size > 2000 and aspect >= digit_one_aspect - aspect_buffer and aspect <= digit_one_aspect + aspect_buffer)):
                # Keep track of the height and y position so we can run averages later
                total_digit_height += h
                total_digit_y += y
                potential_digits.append(contour)
            else:
                if self.debug:
                    cv2.rectangle(self.img, (x, y), (x + w, y + h), (0, 0, 255), 2)

        avg_digit_height = 0
        avg_digit_y = 0
        potential_digits_count = len(potential_digits)
        left_most_digit = 0
        right_most_digit = 0
        digit_x_positions = []
        # Calculate the average digit height and y position so we can determine what we can throw out
        if potential_digits_count > 0:
            avg_digit_height = float(total_digit_height) / potential_digits_count
            avg_digit_y = float(total_digit_y) / potential_digits_count
            #if self.debug:
            #    print("Average Digit Height and Y: " + str(avg_digit_height) + " and " + str(avg_digit_y))

        output = ''
        ix = 0
        # Loop over all the potential digits and see if they are candidates to run through KNN to get the digit
        for id in range(0,len(potential_digits)):
            pot_digit = potential_digits[id]
            id=id+1
            [x, y, w, h] = cv2.boundingRect(pot_digit)
            #print("VALUES: "+str(x)+", "+str(y)+", "+str(w)+", "+str(h))
            # Does this contour match the averages
	    #print(h)
	    #print(y)
	    #print(avg_digit_height)
	    #print(avg_digit_y)
            if h <= avg_digit_height * 1.2 and h >= avg_digit_height * 0.2 and y <= avg_digit_y * 1.2 and y >= avg_digit_y * 0.2:
                # Crop the contour off the eroded image
                cropped = eroded[y:y + h, x: x + w]
                # Draw a rect around it
                cv2.rectangle(self.img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                debug_images.append(('digit' + str(ix), cropped))

                # Call into the KNN to determine the digit
                digit = self.predict_digit(cropped)
                if self.debug:
                    print("Digit: " + digit)
                output += digit

                # Helper code to write out the digit image file for use in KNN training
                if self.write_digits:
                    crop_file_path = CROP_DIR + '/' + digit + '_nat100_' + self.file_name + '_crop_' + str(ix) + '.png'
                    cv2.imwrite(crop_file_path, cropped)

                # Track the x positions of where the digits are
                if left_most_digit == 0 or x < left_most_digit:
                    left_most_digit = x

                if right_most_digit == 0 or x > right_most_digit:
                    right_most_digit = x + w

                digit_x_positions.append(x)

                ix += 1
            else:
                if self.debug:
                    cv2.rectangle(self.img, (x, y), (x + w, y + h), (66, 146, 244), 2)

        decimal_x = 0
        # Loop over the potential digits and find a square that's between the left/right digit x positions on the
        # lower half of the screen
        for pot_decimal in potential_decimals:
            [x, y, w, h] = cv2.boundingRect(pot_decimal)

            if x < right_most_digit and x > left_most_digit and y > (self.height / 2):
                cv2.rectangle(self.img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                decimal_x = x

        # Once we know the position of the decimal, we'll insert it into our string
        for ix, digit_x in enumerate(digit_x_positions):
            if digit_x > decimal_x:
                # insert
                output = output[:ix] + '.' + output[ix:]
                break

        return output

    # Predict the digit from an image using KNN
    def predict_digit(self, digit_mat):
        # Resize the image
        imgROIResized = cv2.resize(digit_mat, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))
        # Reshape the image
        npaROIResized = imgROIResized.reshape((1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))
        # Convert it to floats
        npaROIResized = np.float32(npaROIResized)
        _, results, neigh_resp, dists = self.knn.findNearest(npaROIResized, k=3)
        predicted_digit = str(chr(int(results[0][0])))
        if predicted_digit == 'A':
            predicted_digit = '.'
        return predicted_digit

