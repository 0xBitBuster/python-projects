import numpy as np
import cv2 as cv

# Load paths to model and kernel files
prototxt_path = 'models/colorization_deploy_v2.prototxt'
model_path = 'models/colorization_release_v2.caffemodel'
kernel_path = 'models/pts_in_hull.npy'
image_path = 'images/lion.jpg'

# Initialize the neural network with the Caffe model and load the kernel data
net = cv.dnn.readNetFromCaffe(prototxt_path, model_path)
points = np.load(kernel_path)

# Prepare and assign the kernel data to the model layers
points = points.transpose().reshape(2, 313, 1, 1)
net.getLayer(net.getLayerId("class8_ab")).blobs = [points.astype(np.float32)]
net.getLayer(net.getLayerId("conv8_313_rh")).blobs = [np.full([1, 313], 2.606, dtype='float32')]

# Load the black and white image, normalize it, and convert it to LAB color space
bw_image = cv.imread(image_path)
normalized = bw_image.astype('float32') / 255.0
lab = cv.cvtColor(normalized, cv.COLOR_BGR2LAB)

# Resize the LAB image, extract the L channel, and prepare it for the model
resized = cv.resize(lab, (224, 224))
L = cv.split(resized)[0]
L -= 50

# Perform forward pass to get the colorized image
net.setInput(cv.dnn.blobFromImage(L))
ab = net.forward()[0, :, :, :].transpose((1, 2, 0))

# Resize colorized image to match the original image size
ab = cv.resize(ab, (bw_image.shape[1], bw_image.shape[0]))
L = cv.split(lab)[0]

# Concatenate the L channel and the colorized image to form the final colorized image
colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)
colorized = cv.cvtColor(colorized, cv.COLOR_LAB2BGR)
colorized = (255.0 * colorized).astype('uint8')

# Display the original image and the colorized image
cv.imshow("BW Image", bw_image)
cv.imshow("Colorized Image", colorized)
cv.waitKey(0)
cv.destroyAllWindows()
