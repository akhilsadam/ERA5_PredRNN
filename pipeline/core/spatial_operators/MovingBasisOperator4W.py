__author__ = 'asadam'

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_harmonics as th




# SPECTRAL POD-DMD

class Operator(nn.Module):
    def __init__(self, nlatent, ntime, h, w, device, n_embd=400, nlayers=1, activation=torch.nn.ReLU(), **kwargs):
        super(Operator, self).__init__()
        
        self.nlatent = nlatent
        self.ntime = ntime
        self.nlayers = nlayers
        self.n_embd = n_embd
        self.h = h
        self.w = w
       
        self.m = ntime - 4
        self.n = ntime - self.m
    
        self.device = device
        
        self.activation = activation
        
        
        self.fft = lambda x : torch.fft.rfft(x,n=self.m,dim=1,norm="forward")
        self.ifft = lambda x : torch.fft.irfft(x,n=self.m,dim=1,norm="forward")
        self.m_fft = self.m // 2 + 1    
        
        
        # self.A = nn.Parameter(torch.eye((self.m_fft*self.n),device=device).reshape(self.m_fft,self.n,self.m_fft,self.n)) # f(out), modes(out), f(in), modes(in)
        # self.A_i = nn.Parameter(torch.eye((self.m_fft*self.n),device=device).reshape(self.m_fft,self.n,self.m_fft,self.n)) # f(out), modes(out), f(in), modes(in)
                
        self.A = nn.Parameter(torch.empty((self.m_fft,self.n,self.m_fft,self.n),device=device)) # f(out), modes(out), f(in), modes(in)
        self.A_i = nn.Parameter(torch.empty((self.m_fft,self.n,self.m_fft,self.n),device=device)) # f(out), modes(out), f(in), modes(in)
        nn.init.xavier_uniform_(self.A.data,gain=0.1)        
        nn.init.xavier_uniform_(self.A_i.data,gain=0.1)  
        
        # self.sht = th.RealSHT(h, w, grid="equiangular").to(device)
        # self.n_modes = 720 # seems to be default?
        # self.isht = th.InverseRealSHT(h, w, lmax=self.n_modes, mmax=self.n_modes+1, grid="equiangular").to(device)

        
        # self.lin = nn.ModuleList([nn.Linear(self.n_embd, self.n_embd) for _ in range(nlayers)])
        
        # self.in_cnn = nn.Conv2d(1, self.nchan, kernel_size=3, stride=1, padding=1, bias=True) # identify momenta
        # self.out_cnn = nn.Conv2d(1, self.nchan, kernel_size=3, stride=1, padding=1, bias=False) # create shifts
        # # the two will be multiplied to get shifts proportional to momenta
        # nn.init.xavier_uniform_(self.in_cnn.weight.data)
        # nn.init.constant_(self.in_cnn.bias.data,1.0) # crossout the latter, instead doing skip for stability # bias momenta to 1 to help push the shifts
        # self.out_cnn.weight.data = nn.Parameter((1/self.nchan) * torch.tensor([[0,0,0],[0,1,0],[0,0,0]],device=device)[None,None,:,:].expand(self.nchan,1,3,3))
        # nn.init.xavier_uniform_(self.out_cnn.weight.data)
        # nn.init.constant_(self.out_cnn.bias.data,0.0)
        
        # self.E = nn.Parameter(torch.empty((self.m, self.m, self.n_embd),device=device))
        # self.D = nn.Parameter(torch.empty((self.m, self.m, self.n_embd),device=device))
        # nn.init.xavier_uniform_(self.E)
        # nn.init.xavier_uniform_(self.D)
        
    @torch.compile(fullgraph=False)
    def POD(self,x):
        with torch.no_grad():            
            vssv = torch.einsum("Bt..., BT... -> BtT", x, x) #* (1/(self.ntime-1)) # X^T X, shape BtT, results in vs^t s v^t
            _, e, vt = torch.linalg.svd(vssv)
            s = torch.sqrt(e)
            a2 = torch.where(s > 1, 0, 1)
            d = s / (s**2 + a2)
            sinv = torch.diag_embed(d,offset=0,dim1=-2,dim2=-1)
            u = torch.einsum("Bt..., BTt, BTm -> Bm...", x, vt, sinv)
        return u
    
    @torch.compile(fullgraph=False)
    def batch_POD(self,x):
        # extra batch dim b
        with torch.no_grad():            
            vssv = torch.einsum("Bbt..., BbT... -> BbtT", x, x) #* (1/(self.ntime-1)) # X^T X, shape BtT, results in vs^t s v^t
            _, e, vt = torch.linalg.svd(vssv)
            s = torch.sqrt(e)
            a2 = torch.where(s > 1, 0, 1)
            d = s / (s**2 + a2)
            sinv = torch.diag_embed(d,offset=0,dim1=-2,dim2=-1)
            u = torch.einsum("Bbt..., BbTt, BbTm -> Bbm...", x, vt, sinv)
        return u    
    
    @torch.compile(fullgraph=False)
    def batch_CPOD(self,x):
        u_real = self.batch_POD(x.real)
        u_im = self.batch_POD(x.imag)
        return torch.complex(u_real,u_im)

    @torch.compile(fullgraph=False)
    def liftstep(self,x_fft_chunked,u_fft_chunked):
        # lift
        xhat_fft = torch.einsum("Bfchw,Bfmchw->Bfm",x_fft_chunked[:,:,-1],u_fft_chunked) # take last step
        # step
        cA = torch.complex(self.A,self.A_i)
        yhat_fft = torch.einsum("FMfm,Bfm->BFM",cA,xhat_fft) + xhat_fft # adding skip connection outside parameter
        # reco
        y_fft = torch.einsum("BfM,BfMchw->Bfchw",yhat_fft,u_fft_chunked)
        return y_fft

    # @torch.compile(fullgraph=False)
    def forward(self, x):
        # shape BTchw (T ~= 2*t)
        xu = torch.mean(x,dim=1,keepdim=True)
        x = x - xu
        # Btchw -> Bfchw for each of size-m (t) slices -> Bftchw
        with torch.no_grad():
            x_fft_chunked = torch.stack([self.fft(x[:,i:i+self.m]) for i in range(self.ntime - self.m)],dim=2) 
            # batch POD over each frequency
            u_fft_chunked = self.batch_CPOD(x_fft_chunked) # BfMchw # now M is actual modes!
        
        y_fft = self.liftstep(x_fft_chunked,u_fft_chunked)
                
        # return to realspace
        y = self.ifft(y_fft) # Btchw
        
        # add cohesion loss
        if self.training:
            loss = torch.nn.functional.mse_loss(y[:,:-1],x[:,-self.m+1:])
            loss.backward(retain_graph=True) # only if training!
        
        return y[:,-1:] + xu

    
    
    