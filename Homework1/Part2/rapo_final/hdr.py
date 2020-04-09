import numpy as np


def linearWeight(pixel_value):
    
    z_min, z_max = 0., 255.
    if pixel_value > (z_min+z_max)/2:
        pixel_value = z_max - pixel_value
    else:
        pixel_value = pixel_value-z_min
    
    return pixel_value


def sample_rgb_images(images):
    
    num_intensities = 256
    num_images = len(images)
    intensity_values = np.zeros((num_intensities, num_images), dtype=np.uint8)

    mid_img = images[num_images // 2]

    for zi in range(num_intensities):
        r_idx, c_idx = np.where(mid_img==zi)
        if len(r_idx) > 0:
            random_sample = np.random.randint(0,len(r_idx))
            for zj in range(num_images):
                row_select, col_select = r_idx[random_sample], c_idx[random_sample]
                intensity_values[zi, zj] = images[zj][row_select, col_select]
    
    return intensity_values


def gsolve(intensity_samples, log_exposures, smoothing_lambda, weighting_function): # as in debevec '97
    
    intensity_range = 255
    num_samples = intensity_samples.shape[0]
    num_images = len(log_exposures)

    A = np.zeros((num_images * num_samples + intensity_range,
                      num_samples + intensity_range + 1), dtype=np.float64)
    b = np.zeros((A.shape[0], 1), dtype=np.float64)
    
    # data fitting equation
    
    k = 0
    for i in range(num_samples):
        for j in range(num_images):
            zij = intensity_samples[i, j]
            wij = weighting_function(zij)
            A[k,zij] = wij
            A[k, num_samples+i] = -wij
            b[k,0] = wij*log_exposures[j]
            k += 1
    
    #smoothness equation
    
    for zk in range(1,intensity_range):
        wk = weighting_function(zk)
        A[k, zk-1] = wk * smoothing_lambda
        A[k, zk] = -2 * wk * smoothing_lambda
        A[k, zk+1] = wk * smoothing_lambda
        k += 1
        
    #fix curve - set middle value to 0
    
    A[-1,intensity_range//2] = 1
    
    #svd
    
    mat_A_inv = np.linalg.pinv(A)
    x = np.dot(mat_A_inv, b)

    g = x[0:intensity_range + 1]
    #lE = x[n+1: x.shape[0]]
    
    return g[:, 0]


def compute_hdr_map(images, log_exposure_times, response_curve, weighting_function):
    
    shapes = images[0].shape
    img_rad_map = np.zeros(shapes, dtype=np.float64)

    g = np.zeros(len(images))
    w = np.zeros(len(images))
    
    for i in range(shapes[0]):
        for j in range(shapes[1]):
            for k in range(len(images)):
                g[k] = response_curve[images[k][i,j]]
                w[k] = weighting_function(images[k][i,j])
            Sum_w = np.sum(w)
            if Sum_w>0:
                img_rad_map[i,j] = np.sum(w*(g-log_exposure_times)/Sum_w)
            else:
                img_rad_map[i,j] = g[len(images)//2] - log_exposure_times[len(images)//2]
    
    return img_rad_map

if __name__ == '__main__':
    pass