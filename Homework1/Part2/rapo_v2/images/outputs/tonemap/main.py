
import cv2
import numpy as np

import os
import errno

from os import path

import hdr as hdr


# Change the source folder and exposure times to match your own
# input images. Note that the response curve is calculated from
# a random sampling of the pixels in the image, so there may be
# variation in the output even for the example exposure stack
# SRC_FOLDER = "images/source/sample"
# EXPOSURE_TIMES = np.float64([1 / 160.0, 1 / 125.0, 1 / 80.0,
#                              1 / 60.0, 1 / 40.0, 1 / 15.0])
#
# OUT_FOLDER = "images/output"

SRC_FOLDER = "images"
EXPOSURE_TIMES = np.float64([1. / t for t in [0.03125, 0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]])

OUT_FOLDER = "images/outputs"

EXTENSIONS = set(["bmp", "jpeg", "jpg", "png", "tif", "tiff"])

def countTonemap(hdr_image, min_fraction=0.0005):
    counts, ranges = np.histogram(hdr_image, 256)
    min_count = min_fraction * hdr_image.size
    delta_range = ranges[1] - ranges[0]

    image = hdr_image.copy()
    for i in range(len(counts)):
        if counts[i] < min_count:
            image[image >= ranges[i + 1]] -= delta_range
            ranges -= delta_range

    return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)

def computeHDR(images, log_exposure_times, smoothing_lambda=100.):
    
    images = [np.atleast_3d(i) for i in images]
    num_channels = images[0].shape[2]

    hdr_image = np.zeros(images[0].shape, dtype=np.float64)

    for channel in range(num_channels):

        # Collect the current layer of each input image from
        # the exposure stack
        layer_stack = [img[:, :, channel] for img in images]

        # Sample image intensities
        intensity_samples = hdr.sampleIntensities(layer_stack)

        # Compute Response Curve
        response_curve = hdr.computeResponseCurve(intensity_samples,
                                                  log_exposure_times,
                                                  smoothing_lambda,
                                                  hdr.linearWeight)

        # Build radiance map
        img_rad_map = hdr.computeRadianceMap(layer_stack,
                                             log_exposure_times,
                                             response_curve,
                                             hdr.linearWeight)

        # We don't do tone mapping, but here is where it would happen. Some
        # methods work on each layer, others work on all the layers at once;
        # feel free to experiment.  If you implement tone mapping then the
        # tone mapping function MUST appear in your report to receive
        # credit.
        out = np.zeros(shape=img_rad_map.shape, dtype=img_rad_map.dtype)
        cv2.normalize(img_rad_map, out, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        hdr_image[..., channel] = img_rad_map  
    
    
    
    
    print("hdr pre tonemap")
    #print(hdr_image[:,:,0])
    
    out = countTonemap(hdr_image)
    
    """
    out = np.zeros(shape=img_rad_map.shape, dtype=img_rad_map.dtype)
    tonemap = cv2.createTonemap(2.2)
    ldr = tonemap.process(hdr_image)
    
    print("ldr after tonemap")
    print(ldr[..., 0])
    
    cv2.normalize(ldr, out, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    
        out = np.zeros(shape=img_rad_map.shape, dtype=img_rad_map.dtype)
        cv2.normalize(img_rad_map, out, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        hdr_image[..., channel] = out
    
    print(out[..., 0])
    return out
    """
    #out = np.zeros(shape=hdr_image.shape, dtype=hdr_image.dtype)
    #cv2.normalize(hdr_image, out, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    #hdr_image = out
    
    print("ldr after tonemap")
    #print(out[..., 0])
        
    return out


def main(image_files, output_folder, exposure_times, resize=False):
    """Generate an HDR from the images in the source folder """

    # Print the information associated with each image -- use this
    # to verify that the correct exposure time is associated with each
    # image, or else you will get very poor results
    print("{:^30} {:>15}".format("Filename", "Exposure Time"))
    print("\n".join(["{:>30} {:^15.4f}".format(*v)
                     for v in zip(image_files, exposure_times)]))

    img_stack = [cv2.imread(name) for name in image_files
                 if path.splitext(name)[-1][1:].lower() in EXTENSIONS]

    if any([im is None for im in img_stack]):
        raise RuntimeError("One or more input files failed to load.")

    # Subsampling the images can reduce runtime for large files
    if resize:
        img_stack = [img[::4, ::4] for img in img_stack]

    log_exposure_times = np.log(exposure_times)
    hdr_image = computeHDR(img_stack, log_exposure_times)
    cv2.imwrite(path.join(output_folder, "output.png"), hdr_image)

    print("Done!")


if __name__ == "__main__":
    """Generate an HDR image from the images in the SRC_FOLDER directory """

    np.random.seed()  # set a fixed seed if you want repeatable results

    src_contents = os.walk(SRC_FOLDER)
    dirpath, _, fnames = next(src_contents)

    image_dir = os.path.split(dirpath)[-1]
    print(image_dir)
    output_dir = OUT_FOLDER
    print(output_dir)
    try:
        os.makedirs(output_dir)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

    print("Processing '" + image_dir + "' folder...")

    image_files = sorted([os.path.join(dirpath, name) for name in fnames
                          if not name.startswith(".")])

    main(image_files, output_dir, EXPOSURE_TIMES, resize=False)