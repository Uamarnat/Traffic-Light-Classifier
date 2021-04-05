# # Traffic Light Classifier
# ---

# Your complete traffic light classifier should have:
# 1. **Greater than 90% accuracy**
# 2. ***Never* classify red lights as green**


# # 1. Loading and Visualizing the Traffic Light Dataset

import cv2 # computer vision library
import helpers # helper functions

import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg # for loading in images

get_ipython().run_line_magic('matplotlib', 'inline')

# Image data directories
IMAGE_DIR_TRAINING = "traffic_light_images/training/"
IMAGE_DIR_TEST = "traffic_light_images/test/"


# ## Load the datasets
# Using the load_dataset function in helpers.py
# Load training data
IMAGE_LIST = helpers.load_dataset(IMAGE_DIR_TRAINING)


# ## Visualize the Data
## TODO: Write code to display an image in IMAGE_LIST (try finding a yellow traffic light!)
## TODO: Print out 1. The shape of the image and 2. The image's label

# The first image in IMAGE_LIST is displayed below (without information about shape or label)
selected_image = IMAGE_LIST[50][0]
plt.imshow(selected_image)


# # 2. Pre-process the Data
#### (IMPLEMENTATION): Standardize the input images
# * Resize each image to the desired input size: 32x32px.
# This function should take in an RGB image and return a new, standardized version
def standardize_input(image):
    
    ## TODO: Resize image and pre-process so that all "standard" images are the same size  
    standard_im = np.copy(image)
    standardized_im = cv2.resize(image, (32, 32))
    
    return standardized_im
    


# ## Standardize the output
## Given a label - "red", "green", or "yellow" - return a one-hot encoded label

# Examples: 
# one_hot_encode("red") should return: [1, 0, 0]
# one_hot_encode("yellow") should return: [0, 1, 0]
# one_hot_encode("green") should return: [0, 0, 1]

def one_hot_encode(label):
    
    ## TODO: Create a one-hot encoded label that works for all classes of traffic lights
    one_hot_encoded = [] 
    if label == 'red':
        one_hot_encoded = [1, 0, 0]
    elif label == 'yellow':
        one_hot_encoded = [0, 1, 0] 
    elif label == 'green':
        one_hot_encoded = [0, 0, 1] 
    return one_hot_encoded


# ### Testing as you Code
# Importing the tests
import test_functions
tests = test_functions.Tests()

# Test for one_hot_encode function
tests.test_one_hot(one_hot_encode)


# ## Construct a `STANDARDIZED_LIST` of input images and output labels.

def standardize(image_list):
    
    # Empty image data array
    standard_list = []

    # Iterate through all the image-label pairs
    for item in image_list:
        image = item[0]
        label = item[1]

        # Standardize the image
        standardized_im = standardize_input(image)

        # One-hot encode the label
        one_hot_label = one_hot_encode(label)    

        # Append the image, and it's one hot encoded label to the full, processed list of image data 
        standard_list.append((standardized_im, one_hot_label))
        
    return standard_list

# Standardize all training images
STANDARDIZED_LIST = standardize(IMAGE_LIST)

## TODO: Display a standardized image and its label
plt.imshow(STANDARDIZED_LIST[6][0])


# # 3. Feature Extraction

# Convert and image to HSV colorspace
# Visualize the individual color channels

image_num =49
test_im = STANDARDIZED_LIST[image_num][0]
test_label = STANDARDIZED_LIST[image_num][1]

# Convert to HSV
hsv = cv2.cvtColor(test_im, cv2.COLOR_RGB2HSV)

# Print image label
print('Label [red, yellow, green]: ' + str(test_label))

# HSV channels
h = hsv[:,:,0]
s = hsv[:,:,1]
v = hsv[:,:,2]

# Plot the original image and the three channels
f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20,10))
ax1.set_title('Standardized image')
ax1.imshow(test_im)
ax2.set_title('H channel')
ax2.imshow(h, cmap='gray')
ax3.set_title('S channel')
ax3.imshow(s, cmap='gray')
ax4.set_title('V channel')
ax4.imshow(v, cmap='gray')



# ### (IMPLEMENTATION): Create a brightness feature that uses HSV color space

## TODO: Create a brightness feature that takes in an RGB image and outputs a feature vector and/or value
## This feature should use HSV colorspace values
def create_feature(rgb_image):
    image=np.copy(rgb_image)
    ## TODO: Convert image to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    ## TODO: Create and return a feature value and/or vector
    mask = cv2.inRange(hsv[:,:,2], 5,160)
    masked_image = np.copy(image)
    masked_image[np.where(mask!=0)] = 0
#     masked_image[mask != 0] = [0, 0, 0]
    image = masked_image[2:30,10:25, :]
    plt.imshow(image)
    return image

# (Optional) Add more image analysis and create more features
def check_inrange(pixel, lower, upper):
    """
    Check if H, S, V values of pixel are in between respective range of lower to upper
    """
    yes=[]
    for i in range(len(pixel)):
        if lower[i]<= pixel[i] <= upper[i]:
            yes.append(True)
        else:
            yes.append(False)
    if all(yes):
        return True
    else:
        return False
        
def check_color(masked_image):
    
    #change image to HSV
    image=np.copy(masked_image)
    masked_image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
#     print(masked_image_hsv[11][4], np.array([0,50,20]))
#     print(type(masked_image_hsv[0][0]), type(np.array([0,50,20])))
    
    #split image into 3 sections 
    sections=[masked_image_hsv[0:10],masked_image_hsv[10:20], masked_image_hsv[20:32]]
    
    #check total number of non black pixels in section
    non_black_pixels=[]
    for section in sections:
        count=0
        for row in section:
            for pixel in row:
                if pixel[2]!=0:
                    count +=1
        non_black_pixels.append(count)
    
    #check if pixel is in color range red, yellow, green 
    color_pixel=[0,0,0] #total count of colored pixels in each section
    lower_red1 = np.array([0,10,20]) 
    upper_red1 = np.array([10,255,255])
    lower_red2 = np.array([165,10,20]) 
    upper_red2 = np.array([180,255,255])
    lower_yellow = np.array([15,10,20]) 
    upper_yellow = np.array([32,255,255])
    lower_green = np.array([33,10,20]) 
    upper_green = np.array([90,255,255])
    for i in range(len(sections)):
        color_pixel_count=0
        for row in sections[i]:
            if i ==0:
                for pixel in row:
                    if check_inrange(pixel,lower_red1,upper_red1) or check_inrange(pixel,lower_red2,upper_red2) and pixel[2] !=0:
                        color_pixel_count+=1
                color_pixel[i]=color_pixel_count
            elif i ==1:
                for pixel in row:
                    if check_inrange(pixel,lower_yellow,upper_yellow) and pixel[2] !=0:
                        color_pixel_count+=1
                color_pixel[i]=color_pixel_count
            elif i ==2:
                for pixel in row:
                    if check_inrange(pixel,lower_green,upper_green) and pixel[2] !=0:
                        color_pixel_count+=1
                color_pixel[i]=color_pixel_count

    #get ratio of color pixel to non black pixel:
    ratio=[]
    for i in range(len(color_pixel)):
        if non_black_pixels[i]==0:
            ratio.append(0)
        else:
            ratio.append(color_pixel[i]/ non_black_pixels[i])

    #return ratio list and color pixel list
    return [ratio, color_pixel]

# This function should take in RGB image input
# Analyze that image using your feature creation code and output a one-hot encoded label
def estimate_label(rgb_image):
    
    ## TODO: Extract feature(s) from the RGB image and use those features to
    ## classify the image and output a one-hot encoded label
    get_feature= create_feature(rgb_image)
    classification=check_color(get_feature)
    ratio=classification[0]
    color_pixel=classification[1]
    predicted_label =[0,0,0]
    maxval=ratio.index(max(ratio))
    
    for i in range(3):
        if i==maxval and i == color_pixel.index(max(color_pixel)):
            predicted_label[i]=1
        elif i!=maxval and i == color_pixel.index(max(color_pixel)):
            predicted_label[i]=1
    return predicted_label   


# ## Testing the classifier

# Using the load_dataset function in helpers.py
# Load test data
TEST_IMAGE_LIST = helpers.load_dataset(IMAGE_DIR_TEST)

# Standardize the test data
STANDARDIZED_TEST_LIST = standardize(TEST_IMAGE_LIST)

# Shuffle the standardized test data
random.shuffle(STANDARDIZED_TEST_LIST)


# ## Determine the Accuracy

# Constructs a list of misclassified images given a list of test images and their labels
# This will throw an AssertionError if labels are not standardized (one-hot encoded)

def get_misclassified_images(test_images):
    # Track misclassified images by placing them into a list
    misclassified_images_labels = []

    # Iterate through all the test images
    # Classify each image and compare to the true label
    for image in test_images:

        # Get true data
        im = image[0]
        true_label = image[1]
        assert(len(true_label) == 3), "The true_label is not the expected length (3)."

        # Get predicted label from your classifier
        predicted_label = estimate_label(im)
        assert(len(predicted_label) == 3), "The predicted_label is not the expected length (3)."

        # Compare true and predicted labels 
        if(predicted_label != true_label):
            # If these labels are not equal, the image has been misclassified
            misclassified_images_labels.append((im, predicted_label, true_label))
            
    # Return the list of misclassified [image, predicted_label, true_label] values
    return misclassified_images_labels


# Find all misclassified images in a given test set
MISCLASSIFIED = get_misclassified_images(STANDARDIZED_TEST_LIST)

# Accuracy calculations
total = len(STANDARDIZED_TEST_LIST)
num_correct = total - len(MISCLASSIFIED)
accuracy = num_correct/total

print('Accuracy: ' + str(accuracy))
print("Number of misclassified images = " + str(len(MISCLASSIFIED)) +' out of '+ str(total))

# ### Visualize the misclassified images

# Visualize misclassified example(s)
## TODO: Display an image in the `MISCLASSIFIED` list 
## TODO: Print out its predicted label - to see what the image *was* incorrectly classified as
miss_image_num =1
miss_im = MISCLASSIFIED[miss_image_num][0]
miss_label = MISCLASSIFIED[miss_image_num][1]
true_label = MISCLASSIFIED[miss_image_num][2]
print(miss_label)
print(true_label)
plt.imshow(miss_im)
plt.imshow(create_feature(miss_im))
print(check_color(miss_im))
print(estimate_label(miss_im))

# ## Test if you classify any red lights as green

# Importing the tests
import test_functions
tests = test_functions.Tests()

if(len(MISCLASSIFIED) > 0):
    # Test code for one_hot_encode function
    tests.test_red_as_green(MISCLASSIFIED)
else:
    print("MISCLASSIFIED may not have been populated with images.")


# # 5. Improve your algorithm!