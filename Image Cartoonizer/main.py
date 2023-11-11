import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

#parameters
sigma = 1.2
k = 1.3
threshold = 15
quantization_level=8
input_image_path = "report/data/ankarakalesi.jpg" 

def edgeDedection(input_image, sigma, k, threshold):
    input_image_gray = cv2.cvtColor(input_image, cv2.COLOR_RGB2GRAY)
    
    # equation 1 and 2
    gaussian1 = cv2.GaussianBlur(input_image_gray, (0, 0), sigmaX=sigma)
    gaussian2 = cv2.GaussianBlur(input_image_gray, (0, 0), sigmaX=sigma * k)
    dog_filter = gaussian1 - gaussian2
    # equation 3
    return cv2.threshold(dog_filter, threshold, 255, cv2.THRESH_BINARY)[1]


def smoothImage(input_image, sigma):
    return cv2.GaussianBlur(input_image, (0, 0), sigmaX=sigma)


def quantizeImage(smoothed_image, quantization_level):
    lab_image = cv2.cvtColor(smoothed_image, cv2.COLOR_RGB2LAB)
    L_channel = lab_image[:, :, 0]

    quantized_L_channel = np.floor_divide(L_channel, 256 // quantization_level) * (256 // quantization_level)

    lab_image[:, :, 0] = quantized_L_channel

    quantized_image = cv2.cvtColor(lab_image, cv2.COLOR_LAB2RGB)

    return quantized_image

def cartoonizeImage(input_image):

    #smooth image
    smoothed_image = smoothImage(input_image, sigma)

    #quantization
    quantized_image = quantizeImage(smoothed_image, quantization_level)

    #edge dedection
    edges = edgeDedection(input_image, sigma, k, threshold)
    # Take the inverse of the estimated edges values.
    inverted_edges_image = 255 - edges
    inverted_edges_image = inverted_edges_image[:, :, np.newaxis]

    # Multiply the inverted edges image with the quantized image for each channel.
    return np.multiply(inverted_edges_image, quantized_image)



#------------------main-------------------
#load image
input_image = cv2.imread(input_image_path)
input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

#cartoonize
cartoonized_image = cartoonizeImage(input_image)

#save it to output
# Extract the filename from the input image path
filename = os.path.basename(input_image_path)
filename_without_extension, extension = os.path.splitext(filename)

# Construct the output path based on the filename
output_path = f"report/result/{filename_without_extension}_result{extension}"
cv2.imwrite(output_path, cartoonized_image)
print(f'Resulting image saved at: {output_path}')

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(input_image, cmap='gray')
plt.title("Original Image")
plt.axis('off')


plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(cartoonized_image, cv2.COLOR_BGR2RGB))
plt.title("Cartoon Image")
plt.axis('off')
plt.show()

