import torch
import numpy as np
import logging,os
from tqdm import tqdm
logger = logging.getLogger('preprocessor')

class DataLoader:
    def __init__(self, path, shape):
        self.path = path
        self.shape = shape
    
    def load(self):
        return np.load(self.path)['input_raw_data']

class PreprocessorBase:
    def __init__(self, config):
        
        assert self.datadir is not None, "datadir (directory where data is stored) must be specified"
        assert self.train_data_paths not in [None,[]], "train_data_paths (training datasets) must be specified and not empty"
        assert self.valid_data_paths not in [None,[]], "valid_data_paths (validation datasets) must be specified and not empty"
        assert self.n_var > 0, "n_var (number of variables) must be specified and greater than 0"
        assert self.shapex > 0, "shapex (x dimension of data) must be specified and greater than 0"
        assert self.shapey > 0, "shapey (y dimension of data) must be specified and greater than 0"             

        self.weather_prediction = config['weather_prediction']
        wp = '_WP' if self.weather_prediction else ''
        self.scale_path =  f"{self.datadir}/scale{wp}.txt"
        
        self.input_length = config['input_length']
        self.total_length = config['total_length']
        
        
    def precompute_check(self):
        # check if scale file exists
        if not self.weather_prediction and not os.path.exists(self.scale_path):
            logger.info(f"Scale file {self.scale_path} does not exist! Precomputing...")
            return self.precompute_scale(use_datasets=False)
    
    def precompute_scale(self, use_datasets=False, lazy = False):
        shape = None # (time, var, shapex, shapey)
        datasets = []
        maxs = []
        mins = []
        if not lazy:
            logger.info('Loading training datasets for precomputation (eigenvectors)...')  
            for i,trainset in tqdm(enumerate(self.train_data_paths)):
                data = np.load(trainset, mmap_mode='r')
                try:
                    raw = data['input_raw_data']
                    shape = raw.shape
                    assert len(raw.shape)==4, f"Raw data in {trainset} is not 4D!"
                    assert shape[1] == self.n_var, f"Number of variables in {trainset} does not match n_var!"
                    assert shape[2] == self.shapex, f"Shape of raw data in {trainset} does not match shapex!"
                    assert shape[3] == self.shapey, f"Shape of raw data in {trainset} does not match shapey!"
                    
                    maxs.append(np.nanmax(raw, axis=(0,2,3)))
                    mins.append(np.nanmin(raw,axis=(0,2,3)))                
                    
                except Exception as e:
                    print(f'Warning: Failed to load dataset {i}! Skipping... (Exception "{e}" was thrown.)')
                else:
                    datasets.append(raw)
        
            maxval = np.nanmedian(np.stack(maxs),axis=0)
            minval = np.nanmedian(np.stack(mins),axis=0)
            
            assert np.prod(maxval-minval) > 0, "Max and min values are not valid for at least one dataset!"
            
            scale = 2 / (maxval - minval)
            shift = -1 - minval * scale
            
            # dump scale to file
            np.savetxt(self.scale_path, np.stack([scale,shift]), delimiter=',')
        
        else:
            scale = torch.Tensor([1.0])
            shift = torch.Tensor([0.0])
            import zipfile
            for i,trainset in tqdm(enumerate(self.train_data_paths)):
                try:
                    with zipfile.ZipFile(trainset) as archive:
                        for name in archive.namelist():
                            if name.endswith('input_raw_data.npy'):
                                npy = archive.open(name)
                                version = np.lib.format.read_magic(npy)
                                shape, fortran, dtype = np.lib.format._read_array_header(npy, version)
                                break
                    
                    assert len(shape)==4, f"Raw data in {trainset} is not 4D!"
                    assert shape[1] == self.n_var, f"Number of variables in {trainset} does not match n_var!"
                    assert shape[2] == self.shapex, f"Shape of raw data in {trainset} does not match shapex!"
                    assert shape[3] == self.shapey, f"Shape of raw data in {trainset} does not match shapey!"         
                    
                    raw = DataLoader(trainset,shape)
                    
                except Exception as e:
                    print(f'Warning: Failed to load dataset {i}! Skipping... (Exception "{e}" was thrown.)')
                else:
                    datasets.append(raw)

            
        
        if use_datasets:
            if self.weather_prediction or lazy:
                datasets2 = datasets
            else:
                datasets2 = []
                for data in datasets:
                    datasets2.append(data * scale.reshape((1,len(scale),1,1)) + shift.reshape((1,len(scale),1,1)))
            
            return datasets2, shape, (scale, shift)
        
                
        return shape, (scale, shift)
    
    
    def load_scale(self, device):
        # load scale from text file
    
        if self.weather_prediction:
            scale = torch.Tensor([1.0]).to(device)
            shift = torch.Tensor([0.0]).to(device)
            return scale, shift
    
        try:
            norms = np.loadtxt(self.scale_path, delimiter=',')
            if norms is None:
                # two ints, one per line
                with open(self.scale_path, 'r') as f:
                    scale = float(f.readline())
                    shift = float(f.readline())
            elif isinstance(norms[0], np.float64):
                scale, shift = norms
                scale = torch.Tensor([scale]).to(device)#.unsqueeze(1).unsqueeze(0)
                shift = torch.Tensor([shift]).to(device)#.unsqueeze(1).unsqueeze(0)
            else:
                scale, shift = norms
                scale = torch.from_numpy(scale).to(device)#.unsqueeze(1).unsqueeze(0)
                shift = torch.from_numpy(shift).to(device)#.unsqueeze(1).unsqueeze(0)
            return scale, shift
        except Exception as e:
            print(f'Warning: Failed to load scale file! (Exception "{e}" was thrown.)')
            _, a = self.precompute_scale(use_datasets=False)
            scale = a[0].to('cpu')
            shift = a[1].to('cpu')
            return scale,shift