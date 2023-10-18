import math
import torch
import torch.nn as nn
from core.models.model_base import BaseModel
from core.loss import loss_mixed
import logging
from pydmd import DMD

logger = logging.getLogger("hate")
  
class DMDIntegrator(BaseModel):

    def __init__(self, num_layers, num_hidden, configs):
        # Inheritance
        super(DMDIntegrator, self).__init__(num_layers, num_hidden, configs)
        # self.preprocessor = configs.preprocessor
        self.model_args = configs.model_args

        # Error handling
        # assert self.preprocessor is not None, "Preprocessor is None, please check config! Cannot operate on raw data."
        assert self.model_args is not None, "Model args is None, please check config!"
        assert configs.input_length > 0, "Model requires input_length"
        assert configs.total_length > configs.input_length, "Model requires total_length"
        
        # transformer
        # B S E: batch, sequence, embedding (latent)
        # self.preprocessor.load(device=configs.device)
        self.device = configs.device
        self.input_length = configs.input_length
        self.predict_length = configs.total_length - configs.input_length
        self.total_length = configs.total_length
        self.dmd = DMD(svd_rank=20)

        self.resweight = nn.Parameter(torch.Tensor([0]))

    def edit_config(self,configs):
        if configs.patch_size != 1:
            print(f"Warning: patch_size is {configs.patch_size} but should be 1 for TF model. Setting to 1.")
            configs.patch_size = 1
        return configs
        
        
    def core_forward(self, seq_total, istrain=True, **kwargs):
        dim_b, dim_t, dim_c, dim_x, dim_y = seq_total.shape

        # 1-channel implementation
        out = torch.zeros((dim_b, self.total_length, dim_x, dim_y)).to(self.device).to(torch.cfloat) # Indexed by T
        out[:,:self.input_length,:,:] = seq_total[:,:self.input_length,0,:,:].to(self.device).to(torch.cfloat) # BTCHW
        
        # Iterate over batches
        for i in range(out.shape[0]):
            # Iterate over timesteps
            for j in range(self.input_length, self.total_length):

                logger.info(f"Batch: {i}, Time index: {j}")
                
                # Get current batch up to current time
                data = torch.reshape(out[i,:j,:,:], (j, dim_x * dim_y)) # Reshape into (nt, nx * ny) matrix for DMD

                logger.info(f"Reshaped array of length {j}.")

                # Do the DMD, shake your DMD
                U, S, Vh = torch.linalg.svd(data[:-1,:].T, full_matrices=True)
                Y = data[1:,:].T

                S_pinv = torch.cat((torch.diag(1. / S), torch.zeros(U.shape[1] - Vh.shape[0], Vh.shape[0]).to(self.device).to(torch.cfloat)), dim=0).T

                logger.info(f"SVD completed. Sizes {(U.H.shape, Y.shape, Vh.H.shape, S_pinv.shape)}")

                # Generate the integration matrix
                A = torch.linalg.multi_dot(
                        [U.H, Y, Vh.H, S_pinv]
                    ).to(self.device).to(torch.cfloat)

                logger.info("Integration matrix generated.")

                # Integrate current state to get new state
                inpt = torch.reshape(out[i,j-1,:,:], (dim_x * dim_y,1))
                new_st = (A @ inpt).T # Integrated new state

                logger.info("State integrated.")

                # print(new_st)
 
                # Assign new state to next timestep of current batch
                out[i,j,:,:] = torch.reshape(new_st, (dim_x, dim_y)).T


        # Reshape
        out = torch.real(out.reshape((dim_b, self.total_length, dim_c, dim_x, dim_y)))

        loss_pred = self.resweight
        loss_decouple = torch.tensor(0.0)

        return loss_pred, loss_decouple, out
