import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

#parameters
sigma = 1.1
k = 1.2
threshold = 15
quantization_level=16
input_image_path = "tuna.png" 

def edgeDedection(input_image, sigma, k, threshold):
    input_image_gray = cv2.cvtColor(input_image, cv2.COLOR_RGB2GRAY)

    # equation 1 and 2
    gaussian1 = cv2.GaussianBlur(input_image_gray, (0, 0), sigmaX=sigma)
    gaussian2 = cv2.GaussianBlur(input_image_gray, (0, 0), sigmaX=sigma * k)
    dog_filter = gaussian1 - gaussian2
    # equation 3
    edges = cv2.threshold(dog_filter, threshold, 1, cv2.THRESH_BINARY)[1]

    return edges.astype(np.uint8)


def smoothImage(input_image, sigma):
    return cv2.GaussianBlur(input_image, (0, 0), sigmaX=sigma)


def quantizeImage(smoothed_image, quantization_levels):

    lab_image = cv2.cvtColor(smoothed_image, cv2.COLOR_RGB2LAB)
    L_channel = lab_image[:, :, 0]

    quantized_L_channel = np.floor_divide(L_channel, 256 // quantization_levels) * (256 // quantization_levels)

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
    inverted_edges_image = edges ^ 1

    # Multiply the inverted edges image with the quantized image for each channel.
    return quantized_image * inverted_edges_image[:, :, np.newaxis]



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
plt.imsave(output_path, cartoonized_image, cmap='gray')
print(f'Resulting image saved at: {output_path}')


# Display the cartoon image
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(input_image, cmap='gray')
plt.title("Original Image")
plt.axis('off')


plt.subplot(1, 2, 2)
plt.imshow(cartoonized_image, cmap = 'gray')
plt.title("Cartoon Image")
plt.axis('off')
plt.show()

