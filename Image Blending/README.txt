Images Link: https://drive.google.com/drive/folders/1srS8CS16zPyt1yP7f5PJtN56My24laLI
Login not forced.

Use The Same Image for convinence.

please run the code in command prompt as:

python .\code\main.py or python .\main.py

/------Parameters------/
- number_of_levels = 5
* kernel_size = (5, 5)
- sigma_value = 1.6
- input_image_path = "relativepath.jpg" 
- mask_image_path = "relativepath.jpg" 

/------Functions------/
- def getMask(mask_image):
- def getRegionToMask(input_image):
- def setMaskImage(mask_image, region_to_mask, mask): 
- def downsample(image):
- def upsample(image, target_shape):
- def smooth_image(image):
- def gaussianPyr(image, number_of_levels):
- def laplacianPyr(image, number_of_levels):
- def collapsePyr(laplacian_pyramid):
- def blend(image1, image2, mask_for_blending, number_of_levels):
