import numpy as np
import matplotlib.pyplot as plt
import cv2

kernel_size = (5, 5)
sigma_value = 1.6
number_of_levels = 5

input_image_path = "dog.jpg" 
mask_image_path = "dog.jpg" 

input_image = cv2.imread(input_image_path)
input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
mask_image = cv2.imread(mask_image_path)
mask_image = cv2.cvtColor(mask_image, cv2.COLOR_BGR2RGB)

def getMask(mask_image):
    rect_roi = cv2.selectROI(mask_image)
    cv2.destroyAllWindows()

    return input_image[int(rect_roi[1]):int(rect_roi[1]+rect_roi[3]),
                int(rect_roi[0]):int(rect_roi[0]+rect_roi[2])] 

def getRegionToMask(input_image):
    rect_roi = cv2.selectROI(input_image)
    cv2.destroyAllWindows()

    return rect_roi

def setMaskImage(mask_image, region_to_mask, mask):   

    # create a white image as the size of mask_image
    result_image = np.copy(mask_image)

    # calculate the center of the region_to_mask
    center_x = int(region_to_mask[0] + region_to_mask[2] / 2)
    center_y = int(region_to_mask[1] + region_to_mask[3] / 2)

    # calculate the position to place the mask in the new image
    start_x = center_x - int(mask.shape[1] / 2)
    start_y = center_y - int(mask.shape[0] / 2)

    # create a new image with the mask placed in the specified region
    result_image[start_y:start_y + mask.shape[0], start_x:start_x + mask.shape[1]] = mask

    # create mask for blending image as black and white
    mask_for_blending = np.zeros_like(mask_image, dtype=float)
    mask_for_blending[start_y:start_y + mask.shape[0], start_x:start_x + mask.shape[1]] = 1
    return result_image, mask_for_blending

def downsample(image):
    return image[::2, ::2]

def upsample(image, target_shape):
    return cv2.resize(image, (target_shape[1], target_shape[0]))

def smooth_image(image):
    return cv2.GaussianBlur(image, kernel_size, sigma_value)

def gaussianPyr(image, number_of_levels):
    gaussian_pyramid = [image]
    for i in range(number_of_levels - 1):
        image = smooth_image(image)
        image = downsample(image)
        gaussian_pyramid.append(image)
    return gaussian_pyramid

def laplacianPyr(image, number_of_levels):
    gaussian_pyramid = gaussianPyr(image, number_of_levels)
    laplacian_pyramid = [gaussian_pyramid[number_of_levels - 1]]

    for i in range(number_of_levels - 1, 0, -1):
        expanded = upsample(gaussian_pyramid[i], gaussian_pyramid[i - 1].shape)
        laplacian = gaussian_pyramid[i - 1] - expanded
        laplacian_pyramid.insert(0, laplacian)

    return laplacian_pyramid

def collapsePyr(laplacian_pyramid):
    reconstructed = laplacian_pyramid[-1]

    for laplacian_level in reversed(laplacian_pyramid[:-1]):
        reconstructed = laplacian_level + upsample(reconstructed, laplacian_level.shape)

    return reconstructed

def blend(image1, image2, mask_for_blending, number_of_levels):

    # create laplacian pyramids for both images
    laplacian_pyramid1 = laplacianPyr(image1, number_of_levels)
    laplacian_pyramid2 = laplacianPyr(image2, number_of_levels)

    # Builda Gaussianpyramid for each region mask
    Rgaus = gaussianPyr(mask_for_blending, number_of_levels)

    blended_pyramid = []

    # blend each level of the pyramid using the region mask
    for i in range(len(laplacian_pyramid1)):

        L1 = laplacian_pyramid1[i]
        L2 = laplacian_pyramid2[i]
        R = Rgaus[i]

        # Blend each level of pyramid using region mask from the same level:
        L = L1 * R + L2 * (1 - R)
        blended_pyramid.append(L)

    # Collapse the pyramid to getthe final blended image.
    final_blended_image = collapsePyr(blended_pyramid)

    return final_blended_image, blended_pyramid

# get the mask from the mask image (second image)
mask = getMask(mask_image)

# get the region to mask from input image (first image)
region_to_mask = getRegionToMask(input_image)

image1 = input_image
image2, mask_for_blending = setMaskImage(mask_image, region_to_mask, mask)

mask_for_blending
image1 = image1.astype(np.float64) / 255.0
image2 = image2.astype(np.float64) / 255.0


blended, blended_pyramid = blend(image2,image1,mask_for_blending, number_of_levels)

plt.imshow(blended)
plt.title("Blended Image")
plt.axis('off')
plt.show()