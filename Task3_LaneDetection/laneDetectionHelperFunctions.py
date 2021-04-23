import numpy as np
import cv2
from PIL import Image
import random
import matplotlib.pyplot as plt

def get_label(image, lanes, thickness):
    mask = np.zeros_like(image)
    height = image.shape[0]
    width = image.shape[1]
    colors = [[255,0,0],[0,255,0],[0,0,255],[0,255,255], [255,0,255]]

    for i in range(len(lanes)):
        cv2.polylines(mask, np.int32([lanes[i]]), isClosed=False,color=colors[i], thickness=thickness)
    label = np.zeros((height,width,1),dtype = np.uint8)
    for i in range(len(colors)):
        label[np.where((mask == colors[i]).all(axis = 2))] = 255
    return label

# This function is used to extract lane lines from external data
def get_lanes_from_external(external_labels):
    height = external_labels.shape[0] - 5
    width = external_labels.shape[1]
    lanes = [[],[]]
    while height >= 10:
        left = 0
        right = width - 1
        while left < width:
            if external_labels[height,left] != 0.0:
                lanes[0].append((left, height))
                break
            left += 1
        while right > 0:
            if external_labels[height,right] != 0.0:
                lanes[1].append((right, height))
                break
            right -= 1

        height -= 2
    return lanes

def reshape_2D_to_3D(img):
    reshape_img = np.zeros((80, 160, 1))
    reshape_img[:, :, 0] = img[:,:]
    return reshape_img


def visualize_data_and_label(train_images, labels):
    plt.figure(figsize=(15,15))
    i = 0
    while i < 36:
        plt.subplot(6,6,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        image_index = random.randint(0, len(train_images))
        plt.imshow(train_images[image_index])
        i += 1
        plt.subplot(6,6,i+1)
        plt.imshow(labels[image_index])
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        i += 1
    plt.show()
    
def pre_process(image):
    grayscaled = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gaussianBlur = cv2.GaussianBlur(grayscaled, (5, 5), 0)   
    edgeDetectedImage = cv2.Canny(gaussianBlur, 50, 70)
    return edgeDetectedImage

def region_of_interest(img, vertices):
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with 
    #depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image



def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), 
              minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((*img.shape, 3), dtype=np.uint8)
    
    draw_lines(line_img, lines)
    return line_img

def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

# Post process the prediction image

def predict(model, image):
    reshape_image = image[None,:,:,:]
    prediction = model.predict(reshape_image)
    return prediction[0]

def post_process(prediction):
    filtered_pred = prediction.copy()
    filtered_pred = (prediction[:,:] < 0.04)*1
    filtered_pred = cv2.bitwise_not(filtered_pred)
    filtered_pred += 2
    filtered_pred = filtered_pred.astype('uint8')

    kernel = np.ones((3,3),np.uint8)
    erosion_1 = cv2.erode(filtered_pred,kernel,iterations = 1)
    erosion_1[:,70:90] = 0

    
    # Get and draw houged lines
    rho = 1
    theta = np.pi/180
    threshold = 30
    min_line_len = 5
    max_line_gap = 50
    houged = hough_lines(erosion_1, rho, theta, threshold, min_line_len, max_line_gap)
    houged = cv2.cvtColor(houged, cv2.COLOR_RGB2GRAY)
    
    erosion_2 = cv2.erode(houged,kernel,iterations = 1)
    
    erosion_2[:,70:90] = 0
    return erosion_2
