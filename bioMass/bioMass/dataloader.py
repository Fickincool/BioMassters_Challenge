from skimage import io
import numpy as np
from joblib import Parallel, delayed
import pandas as pd
import os
import tifffile
import numpy as np
from torch.utils.data import Dataset
import torch
from glob import glob
import logging


class SentinelDataset(Dataset):
    '''Sentinel 1 & 2 dataset.'''

    def __init__(self, dir_tiles, dir_target,
                 max_chips=None, transform=None, device='cpu'
                 ):
        '''
        Args:
            dir_tiles -- path to directory containing Sentinel data tiles
            dir_target -- path to directory containing target data (AGWB) tiles
            max_chips -- maximum number of chips to load, used for testing, None --> load all
            transform -- transforms to apply to each sample/batch
            device -- device to load data onto ('cpu', 'mps', 'cuda')
        '''
        
        self.tile_list = glob(dir_tiles+'*')
        self.tile_list = np.unique([x.split('/')[-1].split('_')[0] for x
                                    in self.tile_list if x.endswith('tif')])

        
        if max_chips:
            self.tile_list = self.tile_list[:max_chips]
            
        self.dir_tiles = dir_tiles
        self.dir_target = dir_target
        self.device = device
        self.transform = transform

        # logfile = '/home/ubuntu/Thesis/backup_data/bioMass_data/dataloading_log_test.txt'
        # logging.basicConfig(filename=logfile, level=logging.DEBUG, format="%(asctime)s %(message)s", filemode="w")
        
        return
    
    def __len__(self):
        return len(self.tile_list)
    
    def __getitem__(self, idx):
        chipid = self.tile_list[idx]
        
        # Sentinel 1
        try:
            s1_tile = self._load_sentinel_tiles('S1', chipid)
            s1_tile_scaled = self._scale_channels(s1_tile)
        except Exception as e:
            # logging.debug(f'Data load failure for S1: {chipid}. Exception: {e}')
            print(chipid, e)
            s1_tile_scaled = torch.full([4, 256, 256], torch.nan, dtype=torch.float32, requires_grad=False, device=self.device)
        # Sentinel 2
        try:
            s2_tile = self._load_sentinel_tiles('S2', chipid)
            s2_tile_scaled = self._scale_channels(s2_tile)
        except Exception as e:
            # logging.debug(f'Data load failure for S2: {chipid}. Exception: {e}')
            print(chipid, e)
            s2_tile_scaled = torch.full([11, 256, 256], torch.nan, dtype=torch.float32, requires_grad=False, device=self.device)

        # sentinel_tile = torch.cat([s2_tile_scaled, s1_tile_scaled], axis=0)

        if self.dir_target:
            target_tile = self._load_agbm_tile(chipid)
            # 583.80999756 is the 0.9999 quantile of the whole train data
            target_tile = target_tile.clamp(target_tile.min(), 583.80999756).unsqueeze(0)
            # target_tile = torch.log(target_tile+1)
        else:
            target_tile = torch.full([1, 256, 256], torch.nan, dtype=torch.float32, requires_grad=False, device=self.device)

        sample = {'image_s1': s1_tile_scaled, 'image_s2': s2_tile_scaled,
                  'label': target_tile} # 'image' and 'label' are used by torchgeo

        if self.transform:
            sample = self.transform(sample)

        return sample


#########################################################################################################
    
    def _read_tif_to_tensor(self, tif_path):
        X = tifffile.imread(tif_path).astype(np.float32)
        X = torch.tensor(X,
                         dtype=torch.float32,
                         device=self.device,
                         requires_grad=False,
                         )
        return X

    def _load_sentinel_tiles(self, sentinel_type, chipid):
        file_name = f'{chipid}_{sentinel_type}.tif'
        tile_path = os.path.join(self.dir_tiles, file_name)
        return self._read_tif_to_tensor(tile_path)

    def _load_agbm_tile(self, chipid):
        target_path = os.path.join(self.dir_target,
                                   f'{chipid}_agbm.tif')
        return self._read_tif_to_tensor(target_path)
    
    def _scale_channels(self, x, top_quantile=0.99):
        
        assert len(x.shape)==3
        
        Ms = x.flatten(1, 2).quantile(0.99, dim=1, keepdim=True).unsqueeze(-1)
        ms = torch.amin(x, dim=(1, 2), keepdim=True)
        scaled = (x-ms)/(Ms-ms)
        scaled = scaled.clamp(0,1)
        
        return scaled









data_folder = '/home/ubuntu/Thesis/backup_data/bioMass_data/'

f = os.path.join(data_folder, 'features_metadata.csv')
train_df = pd.read_csv(f)
train_df = train_df[train_df.split=='train']

def read_and_process_tif(filename):
    data = tifffile.imread(filename)
    data = np.float32(data)
    data = np.moveaxis(data, -1, 0)
    # clip top 1% of data ?
    # normalize ?
    
    if '_S2_' in filename:
        nvdi = (data[3]-data[2]) / (data[3]+data[2] + 1e-3) # B5 - B4
        data = np.append(data, nvdi[np.newaxis, ...], axis=0) # band_no: 11

        mndwi = (data[1]-data[3]) / (data[1]+data[3] + 1e-3) # B3 - B5
        data = np.append(data, mndwi[np.newaxis, ...], axis=0) # band_no: 12
        
        evi = 2.5 * data[5] - data[2]/(data[5]+1e-3) + 6 * data[2] - 7.5 * data[0] + 1
        data = np.append(data, evi[np.newaxis, ...], axis=0) # band_no: 13
    
    return data

def read_yearly_tiffs(chip_id, satellite):
    
    chip_df = train_df[(train_df.chip_id==chip_id) & (train_df.satellite==satellite)]

    files_list = chip_df.filename.map(lambda x: os.path.join(data_folder, 'train_features/'+x)).values

    months_list = chip_df.filename.map(lambda x: int(x.split('_')[-1].replace('.tif', ''))).values    

    data = Parallel(len(files_list))(delayed(read_and_process_tif)(filename) for filename in files_list)

    agbm_data = chip_df.corresponding_agbm.map(lambda x: os.path.join(data_folder, 'train_agbm/'+x)).unique()
    assert len(agbm_data)==1
    agbm_data = io.imread(agbm_data[0])

    return data, months_list, agbm_data