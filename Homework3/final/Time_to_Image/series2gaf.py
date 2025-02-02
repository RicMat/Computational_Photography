import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm, trange


def GenerateGAF(all_ts, window_size, rolling_length, fname, normalize_window_scaling=1.0, method='summation', scale='[0,1]'):

   
    n = len(all_ts) 
    
    
    moving_window_size = int(window_size * normalize_window_scaling)
    
    
    n_rolling_data = int(np.floor((n - moving_window_size)/rolling_length))
    
  
    gramian_field = []
    
   
    for i_rolling_data in trange(n_rolling_data, desc="Generating...", ascii=True):

        
        start_flag = i_rolling_data*rolling_length
        
       
        full_window_data =  list(all_ts[start_flag : start_flag+moving_window_size])

       
        rescaled_ts = np.zeros((moving_window_size, moving_window_size), float)
        min_ts, max_ts = np.min(full_window_data), np.max(full_window_data)
        if scale == '[0,1]':
            diff = max_ts - min_ts
            if diff != 0:
                rescaled_ts = (full_window_data - min_ts) / diff
        if scale == '[-1,1]':
            diff = max_ts - min_ts
            if diff != 0:
                rescaled_ts = (2 * full_window_data - diff) / diff

    
        rescaled_ts = rescaled_ts[-int(window_size*(normalize_window_scaling-1)):]
        
        # Gramian Angular Matrix
        this_gam = np.zeros((window_size, window_size), float)
        sin_ts = np.sqrt(np.clip(1 - rescaled_ts**2, 0, 1))
        if method == 'summation':
            # cos(x1+x2) = cos(x1)cos(x2) - sin(x1)sin(x2)
            this_gam = np.outer(rescaled_ts, rescaled_ts) - np.outer(sin_ts, sin_ts)
        if method == 'difference':
            # sin(x1-x2) = sin(x1)cos(x2) - cos(x1)sin(x2)
            this_gam = np.outer(sin_ts, rescaled_ts) - np.outer(rescaled_ts, sin_ts)
            
        gramian_field.append(this_gam)

        del this_gam
    
    # Gramian Angular Field
    np.array(gramian_field).dump('%s_gaf.pkl'%fname)
    
    del gramian_field


def PlotHeatmap(all_img, save_dir='output_img'):

   
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    
    total_length = all_img.shape[0]

    
    fname_zero_padding_size = int(np.ceil(np.log10(total_length)))

    # 輸出圖片
    for img_no in trange(total_length, desc="Output Heatmaps...", ascii=True):
        this_fname = str(img_no).zfill(fname_zero_padding_size)
        plt.imshow(all_img[img_no], cmap='hot', interpolation='nearest')
        plt.axis('off')
        plt.savefig("%s/%s.png"%(save_dir, this_fname), bbox_inches='tight', pad_inches=0, transparent=True)
        #plt.clf()


# =============================================================================
# if __name__=='__main__':
# 
#     random_series = np.random.uniform(low=50.0, high=150.0, size=(200,))
# 
#     timeSeries = list(random_series)
#     windowSize = 50
#     rollingLength = 10
#     fileName = 'demo_%02d_%02d'%(windowSize, rollingLength)
#     GenerateGAF(all_ts = timeSeries,
#                 window_size = windowSize,
#                 rolling_length = rollingLength,
#                 fname = fileName,
#                 normalize_window_scaling = 2.0)
# 
#     ts_img = np.load('%s_gaf.pkl'%fileName,allow_pickle=True )
#     PlotHeatmap(ts_img)
# =============================================================================
