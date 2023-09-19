import numpy as np
import torch
import random
import zipfile
import threading
import gc
import uuid
import logging

logging.basicConfig(filename="data_provider.log",
                    filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.DEBUG)

logger = logging.getLogger('CustomDataProvider')
# logger.setLevel(logging.CRITICAL) # to disable logging

def get_shape(path):
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            if name.endswith('input_raw_data.npy'):
                npy = archive.open(name)
                version = np.lib.format.read_magic(npy)
                shape, fortran, dtype = np.lib.format._read_array_header(npy, version)
                break
    return shape

class DataUnit:
    def __init__(self, paths, shapes, prefetch_size, img_channel1, img_layers, total_length, max_batches):
        self.cpath = None
        self.paths = paths
        self.shapes = shapes
        self.prefetch_size = prefetch_size
        self.img_channel1 = img_channel1
        self.img_layers = img_layers
        self.total_length = total_length
        self.max_batches = max_batches
        self.id = str(uuid.uuid4())[:4]
        
        
        # note last batch is skipped
        self.data = torch.empty((prefetch_size * total_length, img_channel1, shapes[0][2], shapes[0][3]), dtype=torch.float32)
        self.start_index = 0
        self.up_index = -total_length # update index (for the next batch)
        
    def get_index(self, gpu_index=0):
        return (self.start_index + gpu_index) % self.max_batches # this is data index, for a gpu index < max_batches.
        
    def connect(self,i):
        # connect data unit to a particular file
        self.cpath=self.paths[i]
        self.clpath=self.cpath.split('/')[-2][:8]
        logger.info(f"\tDataUnit {self.id} connected to {self.clpath} or index {i}")
    
    def allocate(self, k):
        # prefetch all at once, starting from image k... (gpu_index is set to 0 at this point)
        self.start_index = k
        self.data = torch.from_numpy(
            np.load(self.cpath, mmap_mode='r')["input_raw_data"][k:k+self.prefetch_size*self.total_length, self.img_layers, :, :]
            ).to(torch.float32)
        logger.info(f"\tDataUnit {self.id} allocated from {self.clpath}, with start_index {self.start_index}")
        
        gc.collect()
        
    def update(self, gpu_index):
        # update data unit with new data, starting from image gpu_index... when gpu_index % total_length == 0 and this is not immediately after allocate
        set_index = gpu_index - self.total_length
        cu_index = self.get_index(gpu_index) - self.total_length
        torch.roll(self.data,set_index,dims=0)[:self.total_length] = torch.from_numpy(
            np.roll(np.load(self.cpath, mmap_mode='r')["input_raw_data"],cu_index,axis=0)[:self.total_length, self.img_layers, :, :]
            ).to(torch.float32)
        
        logger.info(f"\tDataUnit {self.id} updated from {self.clpath}:{set_index}:{self.total_length+set_index} <- {cu_index}:{cu_index+self.total_length}")
        
    def get(self, gpu_index):
        # get data unit, starting from image gpu_index... when gpu_index % total_length == 0
        logger.info(f"\tDataUnit {self.id} GET from {self.clpath}:{gpu_index}:{self.total_length+gpu_index}")
        return torch.roll(self.data,gpu_index,dims=0)[:self.total_length]
        

def make_valid(paths):
    cp = []
    for i in range(len(paths)):
        if paths[i].endswith('.npz'):
            try:
                get_shape(paths[i])
            except:
                pass
            else:
                cp.append(paths[i])
    return cp


class DataBatch(torch.utils.data.Dataset):
    def __init__(self, name, paths, total_length, img_channel1, img_layers, prefetch_size, batch_size):
        self.paths = make_valid(paths)
        self.total_length = total_length
        self.img_channel1 = img_channel1
        self.img_layers = img_layers
        self.batch_size = batch_size
        self.prefetch_size = prefetch_size
        self.name = name
        
        self.load()
        
        # shape data is [frame, channel, height, width]
        self.max_batches = min([s[0] for s in self.shapes]) - self.total_length + 1 # exclusive
        self.units = [DataUnit(paths, self.shapes, prefetch_size, img_channel1, img_layers, total_length, self.max_batches) for _ in range(self.batch_size)]
        
        self.total = self.max_batches * len(self.paths)
        
        self.unit_selector = 0
        self.gpu_index = 0
        self.step = 0
        self.threads = []
        
    def __len__(self):
        return self.max_batches
    
    def __getitem__(self, dummy_index):        
        # get unit
        unit = self.units[self.unit_selector]
        self.unit_selector = (self.unit_selector + 1) % self.batch_size
        
        # connect and allocate if starting new dataset
        if self.step == 0:
            # get index
            logger.info(f"High = {len(self.paths)}")
            index = torch.randint(low=0, high=len(self.paths), size=(1,)).item()
            # TODO make striated (skip used indices)
            unit.connect(index)
            unit.allocate(index)
            out = unit.get(self.gpu_index)
        else:
            out = unit.get(self.gpu_index)
            
        # step after all units (every batch)
        if self.unit_selector == 0:
            self.gpu_index = (self.gpu_index + 1) % (self.total_length*self.prefetch_size)
            self.step = (self.step + 1) % self.max_batches
            logger.info(f"Step {self.step} of {self.max_batches}")
        
        # update after every thread (every batch, every unit)
        if self.gpu_index % self.total_length == 0 and self.step != 0:
            # print(self.gpu_index // self.total_length + 1)
            # async updating
            logger.info(f"Starting update thread {self.gpu_index // self.total_length + 1}")
            t = threading.Thread(target=unit.update, args=(self.gpu_index,) )
            t.start()
            self.threads.append(t)
            
            
        # wait for update after last thread (last batch, last unit)    
        if self.unit_selector == len(self.units) - 1:    
            if len(self.threads) == self.prefetch_size * self.batch_size:
                # wait for all updates to finish before re-entering loop
                for t in self.threads:
                    t.join()
                self.threads = []
                gc.collect()
                logger.info(f"Finished update thread {self.gpu_index // self.total_length}")
            
        
        return out
        
    def load(self):
        print(f"{self.name}, Loading data from {self.paths}")
        
        self.shapes = [get_shape(p) for p in self.paths]
        self.lengths = [s[0] for s in self.shapes]
        self.starts = np.cumsum(self.lengths)
        
    

class InputHandle:
    def __init__(self, input_param):
        self.paths = input_param['paths']
        self.num_paths = len(input_param['paths'])
        self.name = input_param['name']
        self.input_data_type = input_param.get('input_data_type', 'float32')
        self.output_data_type = input_param.get('output_data_type', 'float32')
        self.batch_size = input_param['minibatch_size']
        self.is_output_sequence = input_param['is_output_sequence']
        self.concurent_step = input_param['concurent_step']
        self.img_layers = input_param['img_layers']
        if input_param['is_WV']:
            # print(f"input_param['img_channel']: {input_param['img_channel']}")
            self.img_channel1 = int(input_param['img_channel']/10.)
        else:
            self.img_channel1 = input_param['img_channel']

        self.total_length = input_param['total_length']
        self.prefetch_size = input_param.get('prefetch_size', 1) # holds CPU pinned memory of size 2x a single GPU minibatch (timeseries sample) per DataUnit
        
        
        self.dataset = DataBatch(self.name, self.paths, self.total_length, self.img_channel1, self.img_layers, self.prefetch_size, self.batch_size)       

    def total(self):
        return self.dataset.total

    def begin(self, do_shuffle=True):
        self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size = self.batch_size, shuffle = do_shuffle)

    def next(self):
        pass

    def no_batch_left(self):
        return self.dataset.step == 0

    def get_batch(self):
        return next(iter(self.dataloader))
