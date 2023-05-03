import numpy as np
import random

class InputHandle:
    def __init__(self, input_param):
        self.paths = input_param['paths']
        self.num_paths = len(input_param['paths'])
        self.name = input_param['name']
        self.input_data_type = input_param.get('input_data_type', 'float32')
        self.output_data_type = input_param.get('output_data_type', 'float32')
        self.minibatch_size = input_param['minibatch_size']
        self.is_output_sequence = input_param['is_output_sequence']
        self.concurent_step = input_param['concurent_step']
        self.img_layers = input_param['img_layers']
        if input_param['is_WV']:
            # print(f"input_param['img_channel']: {input_param['img_channel']}")
            self.img_channel1 = int(input_param['img_channel']/10.)
        else:
            self.img_channel1 = input_param['img_channel']
        self.data = {}
        self.indices = {}
        self.current_position = 0
        self.current_batch_size = 0
        self.current_batch_indices = []
        self.current_input_length = 48
        self.load()

    def load(self):
        print(f"{self.name}, Loading data from {self.paths[0]}")
        dat_1 = np.load(self.paths[0])
        self.data['clips'] = dat_1['clips']
        self.data['dims'] = dat_1['dims']
        self.data['dims'][0][0] = self.img_channel1
        dat1_raw_data = dat_1['input_raw_data'][:,self.img_layers,:,:]
        # try:
        #     dat1_raw_data = dat_1['input_raw_data'][:,self.img_layers,:,:]
        # except:
        #     print('warning: length of image layers is not consistent with the specified image channel! Using default!')
        #     dat1_raw_data = dat_1['input_raw_data']
        print(f"NaN value num: {np.isnan(dat1_raw_data).sum()}")
        self.data['input_raw_data'] = dat1_raw_data[:,:self.img_channel1,:,:]
        if self.num_paths > 1:
            num_clips_1 = dat_1['clips'].shape[1]
            clip_arr = [dat_1['clips']]
            temp_shape = dat_1['input_raw_data'].shape
            input_raw_arr = np.zeros((temp_shape[0]*self.num_paths, self.img_channel1, 
                                      temp_shape[2], temp_shape[3]))
            curr_pos = temp_shape[0]
            # try:
            #     dat1_raw_data = dat_1['input_raw_data'][:,self.img_layers,:,:]
            # except:
            #     print('warning: length of image layers is not consistent with the specified image channel! Using default!')
            #     dat1_raw_data = dat_1['input_raw_data']
            input_raw_arr[:curr_pos,...] = dat1_raw_data[:,:self.img_channel1,:,:]
            for pathi in range(1, self.num_paths):
                #print(num_clips_1)
                print(f"pathi={pathi}, Loading data from {self.paths[pathi]}")
                dat_2 = np.load(self.paths[pathi])
                next_pos = curr_pos + dat_2['input_raw_data'].shape[0]
                temp_clips = dat_2['clips']
                temp_clips[:,:,0] = dat_2['clips'][:,:,0] + num_clips_1*dat_2['clips'][0,0,1]
                if pathi == self.num_paths - 1:
                    temp_clips = temp_clips[:,:-1,:]
                clip_arr.append(temp_clips)
                #print(temp_clips)
                try:
                    dat2_raw_data = dat_2['input_raw_data'][:,self.img_layers,:,:]
                except:
                    print('warning: length of image layers is not consistent with the specified image channel! Using default!')
                    dat2_raw_data = dat_2['input_raw_data']
                input_raw_arr[curr_pos:next_pos,...] = dat2_raw_data[:,:self.img_channel1,:,:]
                num_clips_1 += dat_2['clips'].shape[1]
                curr_pos = next_pos
            self.data['clips'] = np.concatenate(clip_arr, axis=1)
            self.data['input_raw_data'] = input_raw_arr[:next_pos,...]

        # for key in self.data.keys():
        #     print(key)
        #     print(self.data[key].shape)

    def total(self):
        return self.data['clips'].shape[1]

    def begin(self, do_shuffle=True):
        self.indices = np.arange(self.total()-1,dtype="int32")
        # print(f"self.total(): {self.total()}, self.indices: {self.indices}")
        if do_shuffle:
            random.shuffle(self.indices)
        self.current_position = 0
        if self.current_position + self.minibatch_size <= self.total():
            self.current_batch_size = self.minibatch_size
        else:
            self.current_batch_size = self.total() - self.current_position
        self.current_batch_indices = self.indices[self.current_position : self.current_position+self.current_batch_size]


    def next(self):
        self.current_position += self.current_batch_size
        if self.no_batch_left():
            return None
        if self.current_position + self.minibatch_size <= self.total():
            self.current_batch_size = self.minibatch_size
        else:
            return None
        # print(f"current_position: {self.current_position}, current_position + current_batch_size: {self.current_position + self.current_batch_size}")
        self.current_batch_indices = self.indices[self.current_position:self.current_position + self.current_batch_size]

    def no_batch_left(self):
        if self.current_position+self.minibatch_size >= self.total():
            return True
        else:
            return False

    def input_batch(self):
        if self.no_batch_left():
            return None
        input_batch = np.zeros((self.current_batch_size, self.current_input_length) + tuple(self.data['dims'][0])).astype(self.input_data_type)
        for i in range(self.current_batch_size):
            batch_ind = self.current_batch_indices[i]
            begin = self.data['clips'][0, batch_ind, 0]
            end = self.data['clips'][1, batch_ind, 0] + self.data['clips'][0, batch_ind, 1]
            if batch_ind == 60:
                print(f"begin: {begin}, end: {end}")
                print(f"raw_dat shape: {self.data['input_raw_data'].shape}")
            # print(f"batch_ind: {batch_ind}, begin: {begin}, end: {end}")
            data_slice = self.data['input_raw_data'][begin:end, :self.img_channel1, :, :]
            # print(f"data_slice shape: {data_slice.shape}, self.data['input_raw_data'] shape: {self.data['input_raw_data'].shape}")
            input_batch[i, :self.current_input_length, :, :, :] = data_slice
        input_batch = input_batch.astype(self.input_data_type)
        return input_batch

    def get_batch(self):
        input_seq = self.input_batch()
        return input_seq
        
