please run the code in command prompt as:

python .\code\main.py or python .\main.py


/------Parameters------/
sigma = 1.1
k = 1.3
threshold = 15
quantization_level=32
input_image_path = "relativepath.js" 

/------Functions------/
- edgeDedection(input_image, sigma, k, threshold)
- smoothImage(input_image, sigma):
- quantizeImage(smoothed_image, quantization_levels):
- cartoonizeImage(input_image)