import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tqdm import tqdm
import os
import torch
from sklearn.decomposition import PCA
import logging

from joblib import Parallel, delayed

from skimage import io

from bioMass.dataloader import train_df, read_yearly_tiffs

logfile = '/home/ubuntu/Thesis/backup_data/bioMass_data/train_PCA_warm/log.txt'

logging.basicConfig(filename=logfile, level=logging.DEBUG, format="%(asctime)s %(message)s", filemode="a")


def plot_months(data, months_list, band_no, cmap='viridis'):
    
    fig, ax = plt.subplots(3, 4, sharex=True, sharey=True, figsize=(10, 8))
    list(map(lambda axi: axi.set_axis_off(), ax.ravel()))
    plt.tight_layout()


    for month, month_data in zip(months_list, data):
        col = month%4
        row = month//4
        ax[row][col].imshow(month_data[band_no], cmap=cmap)
        
    plt.show()
    
    return

def get_warm_data(data, months_list):
    warm_months = [0, 8, 9, 10, 11] # Sep, May, June, July, August
    
    data_warm = []
    months_list_warm = []
    for m, d in zip(months_list, data):
        if m in warm_months:
            data_warm.append(d)
            months_list_warm.append(m)

    return data_warm, months_list_warm    

def PCA_by_band(data, band_no, n_components=1):
    single_band_year = np.array(data)[:, band_no, :, :]
    single_band_year = single_band_year.reshape((len(single_band_year), 256*256)).T

    pct_zeros = (single_band_year==0).sum()/len(single_band_year.flatten())
    
    if pct_zeros>0.5:
        logging.debug('Band No. %i. Too many zeros!' %band_no)

    if single_band_year.std()>0:
        single_band_year = (single_band_year - single_band_year.mean()) / single_band_year.std()

        pca = PCA(n_components=n_components)

        reduced_single_band = pca.fit_transform(single_band_year).flatten().reshape(256, 256)

        logging.debug('Band No. %i, Explained variance ratio: %.03f' %(band_no, pca.explained_variance_ratio_[0]))

    else:
        reduced_single_band = single_band_year.T[0].flatten().reshape(256, 256)
        logging.debug('Variance was zero!!!')

    return reduced_single_band    

def compute_PCA_data(chipid, satellite):
    output_folder = '/home/ubuntu/Thesis/backup_data/bioMass_data/train_PCA_warm/'
    output_filename = '%s_%s.tif' %(chipid, satellite)
    output_filename = os.path.join(output_folder, output_filename)
    
    if satellite=='S1':
        total_bands = 4
    elif satellite=='S2':
        total_bands = 11
    
    if not os.path.exists(output_filename):
    # if True:

        data, months_list, agbm = read_yearly_tiffs(chipid, satellite)  
        data, months_list = get_warm_data(data, months_list)

        logging.info('Chip ID: %s, Satellite: %s, # Warm months: %i' %(chipid, satellite, len(months_list)))

        if len(months_list)<2:
            logging.info('Not enough data to compute PCA for this Chip ID!!!!')
        
        else:
            reduced_data = [PCA_by_band(data, band_no) for band_no in range(0, total_bands)]

            out_data = np.array(reduced_data)

            io.imsave(output_filename, out_data, plugin='tifffile')
    else:
        pass
    
    return

for satellite in ['S2', 'S1']: 
    for chipid in tqdm(train_df.chip_id.unique()):
        compute_PCA_data(chipid=chipid, satellite=satellite)