import numpy as np
import scipy as sp


def linearWeight(pixel_value):
    
    z_min, z_max = 0., 255.
    # WRITE YOUR CODE HERE.
    if pixel_value > (z_min+z_max)/2:
        pixel_value = z_max - pixel_value
    else:
        pixel_value = pixel_value-z_min
    return pixel_value


def sampleIntensities(images):
    
    # There are 256 intensity values to sample for uint8 images in the
    # exposure stack - one for each value [0...255], inclusive
    num_intensities = 256
    num_images = len(images)
    intensity_values = np.zeros((num_intensities, num_images), dtype=np.uint8)

    # Find the middle image to use as the source for pixel intensity locations
    mid_img = images[num_images // 2]  # using integer division is arbitrary

    
    # WRITE YOUR CODE HERE.
    for zi in range(num_intensities):
        r_idx, c_idx = np.where(mid_img==zi)
        if len(r_idx) > 0:
            random_sample = np.random.randint(0,len(r_idx))
            for zj in range(num_images):
                row_select, col_select = r_idx[random_sample], c_idx[random_sample]
                intensity_values[zi, zj] = images[zj][row_select, col_select]
    return intensity_values


def computeResponseCurve(intensity_samples, log_exposures, smoothing_lambda, weighting_function):
    
    intensity_range = 255  # difference between min and max possible pixel value for uint8
    num_samples = intensity_samples.shape[0]
    num_images = len(log_exposures)

    # NxP + [Zmax - (Zmin + 1)] + 1 constraints; N + 256 columns
    mat_A = np.zeros((num_images * num_samples + intensity_range,
                      num_samples + intensity_range + 1), dtype=np.float64)
    mat_b = np.zeros((mat_A.shape[0], 1), dtype=np.float64)

    # WRITE YOUR CODE HERE
    k = 0
    for i in range(num_samples):
        for j in range(num_images):
            zij = intensity_samples[i, j]
            wij = weighting_function(zij)
            mat_A[k,zij] = wij
            mat_A[k, num_samples+i] = -wij
            mat_b[k,0] = wij*log_exposures[j]
            k += 1

    # WRITE YOUR CODE HERE
    for zk in range(1,intensity_range):
        wk = weighting_function(zk)
        mat_A[k, zk-1] = wk * smoothing_lambda
        mat_A[k, zk] = -2 * wk * smoothing_lambda
        mat_A[k, zk+1] = wk * smoothing_lambda
        k += 1

    # WRITE YOUR CODE HERE
    mat_A[-1,intensity_range//2] = 1
    
    # WRITE YOUR CODE HERE
    mat_A_inv = np.linalg.pinv(mat_A)
    x = np.dot(mat_A_inv, mat_b)
    # raise NotImplementedError


    
    g = x[0:intensity_range + 1]

    return g[:, 0]


def computeRadianceMap(images, log_exposure_times, response_curve, weighting_function):
    
    img_shape = images[0].shape
    img_rad_map = np.zeros(img_shape, dtype=np.float64)

    
    # WRITE YOUR CODE HERE
    g = np.zeros(len(images))
    w = g.copy()
    for i in range(img_shape[0]):
        for j in range(img_shape[1]):
            for k in range(len(images)):
                g[k] = response_curve[images[k][i,j]]
                w[k] = weighting_function(images[k][i,j])
            SumW = np.sum(w)
            #if SumW>0:
            img_rad_map[i,j] = np.sum(w*(g-log_exposure_times)/SumW)
            #else:
                #img_rad_map[i,j] = g[len(images)//2] - log_exposure_times[len(images)//2]
    return img_rad_map

if __name__ == '__main__':
    pass