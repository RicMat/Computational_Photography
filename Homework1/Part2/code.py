import os
import numpy as np
import cv2


folder = 'images'

# We get all the image files from the source folder
files = list([os.path.join(folder, f) for f in os.listdir(folder)])

# We compute the average by adding up the images
# Start from an explicitly set as floating point, in order to force the
# conversion of the 8-bit values from the images, which would otherwise overflow
average = cv2.imread(files[0]).astype(np.float)
for file in files[1:]:
    image = cv2.imread(file)
    # NumPy adds two images element wise, so pixel by pixel / channel by channel
    average += image
 
# Divide by count (again each pixel/channel is divided)
average /= len(files)

# Normalize the image, to spread the pixel intensities across 0..255
# This will brighten the image without losing information
output = cv2.normalize(average, None, 0, 255, cv2.NORM_MINMAX)

# Save the output
cv2.imwrite('output.png', output)

#hdr

def loadExposureSeq(path):
    images = []
    times = []
    with open('list.txt') as f:
        content = f.readlines()
    for line in content:
        tokens = line.split()
        images.append(cv2.imread(os.path.join(path, tokens[0])))
        times.append(1. / float(tokens[1]))
    return images, np.asarray(times, dtype=np.float32)


images, exposures = loadExposureSeq('images')

# Compute the response curve
calibration = cv2.createCalibrateDebevec()
response = calibration.process(images, exposures)

# Compute the HDR image
merge = cv2.createMergeDebevec()
hdr = merge.process(images, exposures, response)

# Save it to disk
cv2.imwrite('hdr_image.hdr', hdr)


tonemap = cv2.createTonemap(2.2)
ldr = tonemap.process(hdr)

merge_mertens = cv2.createMergeMertens()
fusion = merge_mertens.process(images)

# Tonemap operators create floating point images with values in the 0..1 range
# This is why we multiply the image with 255 before saving
cv2.imwrite('durand_image.png', ldr * 255)
cv2.imwrite('fusion.png', fusion * 255)
