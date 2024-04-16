__author__ = 'asadam'

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_harmonics as th
import torchvision.transforms as tf
import math

from tqdm import tqdm

from normalize import norm_scales as scale

pi = math.pi   
rho = 1.204

const_L = 6370*1000.0
const_V = scale[0] # or scale[1]
const_P = scale[3] / (rho * const_V**2) # multiply by this
const_T = const_V / const_L # multiply by this
const_A = 1 / const_T # multiply by this

dt = 6 * 3600.0
omega = 7.3e-5


l_star = 1.0 # by definition
dt_star = dt * const_T
omega_star = omega * const_A

v_clamp = 60

# 2D incompressible N-S

def check_validity(m):
    # BCHW
    with torch.no_grad():
        out = m[0,0,0,0] + m[-1,-1,-1,-1]

def _matrix_pow(matrix: torch.Tensor, p: float) -> torch.Tensor: # not the best, seems to die badly!
    r"""
    Power of a matrix using Eigen Decomposition.
    Args:
        matrix: matrix
        p: power
    Returns:
        Power of a matrix
    """
    vals, vecs = torch.linalg.eig(matrix)
    vals_pow = vals.pow(p)
    matrix_pow = torch.einsum("Bab,Bb,Bbc->Bac",vecs, vals_pow, torch.inverse(vecs))
    return matrix_pow.real

class Operator(nn.Module):
    def __init__(self, nlatent, ntime, h, w, device, n_embd=400, nlayers=1, activation=torch.nn.ReLU(), **kwargs):
        super(Operator, self).__init__()
        
        # print(f"NLATENT:{nlatent}")
        
        self.nlatent = nlatent
        self.clatent = nlatent - 2 # no velocity
        self.ntime = ntime
        self.nlayers = nlayers
        self.n_embd = n_embd
        self.h = h
        self.w = w
    
        self.device = device
        
        self.activation = activation     
        
        self.nlat = h
        self.nlon = w
        d_lat = pi/(2*h)
        # d_lon = pi/w
        self.lat = torch.linspace(-pi/2 + d_lat,pi/2 - d_lat,steps=h, device=device)
        # self.lon = torch.linspace(d_lon,2*pi - d_lon,steps=w, device=device)

        # Step size as a function of latitude (in m)        
        self.l_star = torch.tensor([l_star],device=device)
        self.dx = lambda r: (1/h)*2*pi*r*torch.cos(self.lat) # dx (longitudinal steps along cylindrical perimeter) as a function of latitude
        self.y = lambda r: r*torch.sin(self.lat) # y (latitude) axial height as a function of latitude
        self.dt = torch.tensor([dt_star],device=device) # 6-Hourly resolution # TODO make parameter

        self.wx = lambda r: 2 * omega_star * r * torch.cos(self.lat)
        self.wwx = lambda r: omega_star * omega_star * r * torch.cos(self.lat)
        self.rsin = lambda r: r * torch.sin(self.lat)
        self.rcos = lambda r: r * torch.cos(self.lat)
        # Coriolis parameter
        self.f_star = 2*omega_star*torch.sin(self.lat) # Rotation rate (coriolis, 2w from 2w x v)
        
        
        
        self.rate = nn.Parameter(torch.ones((self.clatent,h,w,2),device=device)) # friction-modulated velocity adjustment
        self.diffusivity = nn.Parameter(torch.zeros((self.clatent),device=device)) # TODO make space-dependent!
        
        # n_modes = ntime-1
        # self.Ax = nn.Parameter(torch.empty((n_modes,n_modes),device=device))
        # nn.init.xavier_uniform_(self.Ax.data,gain=0.001)
        # self.Ay = nn.Parameter(torch.empty((n_modes,n_modes),device=device))
        # nn.init.xavier_uniform_(self.Ay.data,gain=0.001)
        
        # regularization
        # self.f_cohese = 1e-4
        # self.v_cohese = 1e-2
        self.c_cohese = 1e-2
        
        # self.POD_kern = 5 # even

        
        # local (neighbor modes)
        self.neighbor_kern = 8 # this times stride must be less than padding! NOTE: add padding increase eventually to accomodate large stride..
        self.neighbor_stride = 1
        
        assert 'accelerator' in kwargs
        self.accelerator = kwargs['accelerator']
        # CNN Kernel (along x, generated along y)
        # quick MLP
        self.CNN_kern = 21# 21x21 # must be odd
        
        self.pad = (self.CNN_kern)//2
        
        self.CNN_chan = 2
        self.CNN_out_deriv = 12 # 8 output "derivatives"
        self.CNN_out = self.CNN_chan*self.CNN_out_deriv
        # so (8*2) x 2 x 5 x 5
        self.CNN_p = self.CNN_out*self.CNN_chan*self.CNN_kern*self.CNN_kern
        self.p_embd = 100
        self.embd = 40
        
        self.CNN_kernel = nn.Parameter(1/self.CNN_kern * torch.ones((self.CNN_p),device=device))
        pos = 4 # r * sine and cosine of latitude, 2wx, wwx
        self.local_network = nn.Sequential(
          nn.Linear(self.CNN_p+pos,self.p_embd),
          nn.ReLU(),
          nn.Linear(self.p_embd,self.p_embd),
          nn.ReLU(),
          nn.Linear(self.p_embd,self.CNN_p),
        )
        [self.init_weights(k) for k in self.local_network]
        
        self.pos = lambda r: torch.stack([self.rsin(r),self.rcos(r),self.wx(r),self.wwx(r)], dim=-1)
        
        self.CNN_reshape = lambda x: x.reshape(self.CNN_out,self.CNN_chan,self.CNN_kern,self.CNN_kern)
        self.CNN_localizer = lambda k, p: self.CNN_reshape(self.local_network(torch.cat([k,p], dim=-1)) + k) # generates kernel for convolution!
        
        
        self.local_eval = nn.Sequential(
          nn.Linear(self.CNN_out, self.embd),
          nn.ReLU(),
          nn.Linear(self.embd,self.CNN_chan),
        )
        [self.init_weights(k) for k in self.local_eval]
        
        self.CNN = lambda f,k : torch.nn.functional.conv2d(f,k,bias=None,padding='valid') # takes in patch of self.CNN_chan * self.CNN_kern * self.w_pad and returns self.CNN_out * 1 * self.w
        self.loss = torch.nn.MSELoss()
        
        self.print_every = 20
        self.iter = 0

    def apply_CNN_explicit_single_step(self, x_pad, xshape, pos):
        x_hat = torch.zeros(xshape, device=self.device)
        with self.accelerator.no_sync(self):
            for t in range(x_pad.shape[1]): # all times
                for j in range(self.h):
                    jl = j #- self.pad
                    jm = j + self.pad
                    jh = j + self.CNN_kern    
                    
                    patch = x_pad[:,t,...,jl:jh,:].detach()
                    
                    kernel = self.CNN_localizer(self.CNN_kernel,pos[j,:])
                    convolved = self.CNN(patch, kernel).permute(0,2,3,1)
                    # print(convolved.shape)
                    xshift = self.local_eval(convolved).permute(0,3,1,2)
                    
                    x_hat[:,t,...,j,:] = xshift.detach()[None,:,0,:] # TODO fix this weird indexing
                    
                    if t < 10: # training window
                        loss = self.loss(xshift, x_pad[:,t+1,...,jm,self.pad:-self.pad])
                        self.accelerator.backward(loss) 
                        # print(t)   
        # print("okay!!")
        out = x_hat.detach()
        out.requires_grad = True
        return out
        
        
    # @torch.compile(fullgraph=False)
    def apply_CNN_explicit_inner(self, x_hat, pos):
        x_pad = self.pad_in(x_hat[:,None,...]) # make image to sample from

        x_new = []
        for j in range(self.h): # do patches
            jl = j #- self.pad
            jm = j + self.pad
            jh = j + self.CNN_kern    
            
            patch = x_pad[:,0,...,jl:jh,:]
            
            kernel = self.CNN_localizer(self.CNN_kernel,pos[j,:])
            convolved = self.CNN(patch, kernel).permute(0,2,3,1)
            # print(convolved.shape)
            xshift = self.local_eval(convolved).permute(0,3,1,2)
            # print(xshift.shape)
            x_new.append(xshift)
            
        # now update base image
        # print(f"here : {len(x_new)}; {torch.cuda.max_memory_allocated()/10**9}")
        # for i,t in enumerate(x_new):
        #     print(i)
        #     check_validity(t)
        # print("valid!")  
        # q = torch.stack(x_new,dim=0) # success
        # print("test1")
        # q = torch.stack(x_new,dim=-2) # fail
        # print("test2")
        # q = torch.cat(x_new,dim=0) # success
        # print("end tests")                  
        # x_hat_2 = torch.cat(x_new,dim=-2) # along j/h # b,c,h,w # somehow this fails??
        
        x_hat_2 = torch.stack(x_new,dim=0)[...,0,:].permute(1,2,0,3)  # h, b, c, 1, w - > h,b,c,w -> b,c,h,w
        return x_hat_2
        
    def apply_CNN_explicit(self, x, pos, steps=20):
        x_out = []
        for t in range(x.shape[1]): # all times
            x_hat = x[:,t,...].detach().clone() # base image
            x_hat.requires_grad = True
            
            for q in range(steps): # all steps
                x_hat = self.apply_CNN_explicit_inner(x_hat, pos)
                print(f"{q} @ {torch.cuda.max_memory_allocated()/10**9}") 
                # up to 50 steps on a 3x3 forward on Frontera (16GB),
                # if including backward pass, 30 seems to work at a respectable 10.62 GB 
                # 21x21 @ 20 steps = 11.22 GB
                                
            x_out.append(x_hat.detach()) # after all steps, update output list    
            
            # print(f"update @ {torch.cuda.max_memory_allocated()/10**9}") # takes about 6.6 GB for 21x21 at 10 steps
            # after all steps, do loss
            if t < 10: # training window
                loss = self.loss(x_hat, x[:,t+1,...]) # next image
                self.accelerator.backward(loss) 
                    # print(t)   
        # print("okay!!")
        out = torch.stack(x_out, dim=1)
        out.requires_grad = True
        return out
        
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform(m.weight, gain=0.01)
            m.bias.data.fill_(0.0)
    
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

    # @torch.compile(fullgraph=False)
    def sparsePOD(self, x, PVE_cut = 0.99):
        with torch.no_grad():            
            vssv = torch.einsum("Bt..., BT... -> BtT", x, x) #* (1/(self.ntime-1)) # X^T X, shape BtT, results in vs^t s v^t
            _, e, vt = torch.linalg.svd(vssv)
            pe = e.sum(dim=0)
            positive = (torch.cumsum(pe[1:], dim=0) / torch.sum(pe[1:]) > PVE_cut).tolist() # make sure to skip the mean
            n = next((i for i in range(len(positive)) if positive[i]), len(positive))
            # print(f"PVE cut reached at {n}")
            d = 1 / torch.sqrt(e[:,:n])
            sinv = torch.diag_embed(d,offset=0,dim1=-2,dim2=-1)
            u = torch.einsum("Bt..., BTt, BTm -> Bm...", x, vt[:,:n,:], sinv)
        return u
    
    def truncated_POD_snapshot(self, x, PVE_cut = 200):
        with torch.no_grad():            
            vssv = torch.einsum("Bt..., BT... -> BtT", x, x) #* (1/(self.ntime-1)) # X^T X, shape BtT, results in vs^t s v^t
            _, e, vt = torch.linalg.svd(vssv)
            n = PVE_cut
            # print(f"PVE cut reached at {n}")
            d = 1 / torch.sqrt(e[:,:n])
            sinv = torch.diag_embed(d,offset=0,dim1=-2,dim2=-1)
            u = torch.einsum("Bt..., BTt, BTm -> Bm...", x, vt[:,:n,:], sinv)
        return u
    
    def truncated_POD(self, x): # other snapshot style!
        with torch.no_grad():     
            b = x.shape[0]    
            t = x.shape[1]
            x_flat = x.reshape(b,t,-1)
            if x_flat.shape[-1] > t : 
                u_flat = self.truncated_POD_snapshot(x_flat)
            else:
                u_flat = self.POD(x_flat.permute(0,2,1)).permute(0,2,1) # BM(chw)
            u = u_flat.reshape(b,-1,*x.shape[2:])
        return u
    
    def truncated_CPOD(self, x): # other snapshot style!
        with torch.no_grad():     
            b = x.shape[0]    
            t = x.shape[1]
            x_flat = x.reshape(b,t,-1)
            u_flat_real = self.truncated_POD_snapshot(x_flat.real)
            u_flat_imag = self.truncated_POD_snapshot(x_flat.imag)
            u_flat = torch.complex(u_flat_real,u_flat_imag)
            u = u_flat.reshape(b,-1,*x.shape[2:])
        return u

    @torch.compile(fullgraph=False)
    def source_DMD(self,x):
        with torch.no_grad():
            u = self.POD(x) # BM...
            z = torch.einsum("BT..., BM... -> BTM", x, u)
            
            D_plus = torch.linalg.lstsq(z[:,:-1,:],z[:,1:,:]).solution # B,t,m_in  + B,t,m_out -> B,m_in,m_out           # forward DMD (AX-B)
            D_minus = torch.linalg.lstsq(z[:,1:,:],z[:,:-1,:]).solution #  -> B,m_out,m_in (since reverse direction)     # backward DMD
            _pm = torch.einsum("Bba,Bbc->Bca",D_plus,D_minus)
            D_plusminus = _matrix_pow(_pm,0.5)                          # square root of product
            
            
            z_step = torch.einsum("Bc,Bca->Ba",z[:,-1,:],D_plusminus)           # step
            x_new = torch.einsum("Bm, Bm... -> B...", z_step, u)[:,None,...]    # reproject
            # y = torch.cat([x,x_new],dim=1) # B(T+1)...                          # cat
            
        return x_new
    
    @torch.compile(fullgraph=False)
    def reg_DMD(self,x):
        with torch.no_grad():
            u = self.POD(x)
            z = torch.einsum("BT..., BM... -> BTM", x, u)
            
            D_plus = torch.linalg.lstsq(z[:,:-1-4,:],z[:,1:-4,:]).solution # B,t,m_in  + B,t,m_out -> B,m_in,m_out           # forward DMD (AX-B)
            
            z_step = torch.einsum("BTc,Bca->BTa",z,D_plus)           # step
            x_new = torch.einsum("BTm, Bm... -> BT...", z_step, u)    # reproject
            # y = torch.cat([x,x_new],dim=1) # B(T+1)...                          # cat
            
        return x_new
    
    @torch.compile(fullgraph=False)
    def mod_DMD(self,x,u):
        with torch.no_grad():
            z = torch.einsum("BT..., BM... -> BTM", x, u)
            
            D_plus = torch.linalg.lstsq(z[:,:-1-4,:],z[:,1:-4,:]).solution # B,t,m_in  + B,t,m_out -> B,m_in,m_out           # forward DMD (AX-B)
            
            z_step = torch.einsum("BTc,Bca->BTa",z,D_plus)           # step
            x_new = torch.einsum("BTm, Bm... -> BT...", z_step, u)    # reproject
            # y = torch.cat([x,x_new],dim=1) # B(T+1)...                          # cat
            
        return x_new
    
    # @torch.compile(fullgraph=False)
    # def spectral_DMD(self,x):
    #     # Btchw -> Bfchw for each of size-m (t) slices -> Bftchw:
    #     x_fft_chunked = torch.stack([self.fft(x[:,i:i+self.m]) for i in range(self.ntime - self.m)],dim=2) 
    #     # batch POD over each frequency
    #     u_fft_chunked = self.batch_CPOD(x_fft_chunked) # BfMchw # now M is actual modes!
        
    #     y_fft = self.liftstep(x_fft_chunked,u_fft_chunked)
                
    #     # return to realspace
    #     y = self.ifft(y_fft) # Btchw
    
    # @torch.compile(fullgraph=False)
    # def resolvent_DMD(self,x,u):
    #     x_linear = self.mod_DMD(x,u)
    #     forcing = x[:, 1:] - x_linear[:,:-1]
    #     x_forced = self.spectral_DMD(forcing)
            
    #     return x_new
    
    @torch.compile(fullgraph=False)
    def source_DMD_seq(self,x):
        with torch.no_grad():
            u = self.POD(x) # BM...
            z = torch.einsum("BT..., BM... -> BTM", x, u)
            
            D_plus = torch.linalg.lstsq(z[:,:-1,:],z[:,1:,:]).solution # B,t,m_in  + B,t,m_out -> B,m_in,m_out           # forward DMD (AX-B)
            D_minus = torch.linalg.lstsq(z[:,1:,:],z[:,:-1,:]).solution #  -> B,m_out,m_in (since reverse direction)     # backward DMD
            _pm = torch.einsum("Bba,Bbc->Bca",D_plus,D_minus)
            D_plusminus = _matrix_pow(_pm,0.5)                          # square root of product
            
            
            z_step = torch.einsum("BTc,Bca->BTa",z,D_plusminus)           # step
            x_new = torch.einsum("BTm, Bm... -> BT...", z_step, u)
            # y = torch.cat([x,x_new],dim=1) # B(T+1)...                          # cat
            
        return x_new
    
    @torch.compile(fullgraph=False)
    def pad_in(self,x,pad=-1):
        if pad < 0:
            pad = self.pad
        shape = x.shape[:-2]        
        x = x.reshape((x.shape[0]*x.shape[1]*x.shape[2], self.h, self.w))
        x = F.pad(x, (pad,pad), mode='circular')
        x = F.pad(x, (0,0,pad,pad), mode='reflect') # size is (BxTxC)hw, where hw = H+2p, W+2p
        x = x.reshape((*shape, self.h + 2*pad, self.w + 2*pad))
        return x
    
    def window_in_out(self,x_pad):    
        x_hat = torch.zeros_like(x_pad,device=self.device)
        norm = torch.zeros((self.h + 2*self.pad, self.w + 2*self.pad),device=self.device)
        for i in range(0,self.w+self.pad,self.stride):
            il = i #- self.pad
            ih = i + 2*self.pad
            for j in range(0,self.h+self.pad,self.stride):
                jl = j #- self.pad
                jh = j + 2*self.pad    
                
                patch = x_pad[...,jl:jh,il:ih]
                u = self.POD(patch) # BM...
                y = self.reg_DMD(patch,u)
                x_hat[...,jl:jh,il:ih] = x_hat[...,jl:jh,il:ih] + y
                norm[jl:jh,il:ih] = norm[jl:jh,il:ih] + 1.0

        return x_hat / norm
    
    
    def window_in_out_2(self,x_pad):    
        x_hat = torch.zeros_like(x_pad,device=self.device)
        norm = torch.zeros((self.h + 2*self.pad, self.w + 2*self.pad),device=self.device)
        for i in tqdm(range(self.pad,self.w,self.stride)): # skip ends
            il = i #- self.pad
            ih = i + 2*self.pad
            for j in range(self.pad,self.h,self.stride): # skip ends
                jl = j #- self.pad
                jh = j + 2*self.pad    
                   
                u_list = []
                for i_ in range(-self.neighbor_kern,self.neighbor_kern):
                    pil = il + i_*self.neighbor_stride
                    pih = ih + i_*self.neighbor_stride
                    for j_ in range(-self.neighbor_kern,self.neighbor_kern):
                        pjl = jl + j_*self.neighbor_stride
                        pjh = jh + j_*self.neighbor_stride
                        
                        patch_ij = x_pad[...,pjl:pjh,pil:pih]
                        pu = self.POD(patch_ij) # BM...
                        u_list.append(pu)
                        
                u = torch.cat(u_list, dim=1) # cat along mode-dim, since these are all modes!
                u = self.sparsePOD(u)
                patch = x_pad[...,jl:jh,il:ih]
                y = self.mod_DMD(patch,u)
                x_hat[...,jl:jh,il:ih] = x_hat[...,jl:jh,il:ih] + y
                norm[jl:jh,il:ih] = norm[jl:jh,il:ih] + 1.0

        return x_hat / norm
    
    # @torch.compile(fullgraph=False)
    def window_in(self,x):    
        us = []
        zs = []
        for i in range(0,self.w+self.pad,self.stride):
            il = i #- self.pad
            ih = i + 2*self.pad
            us_y = []
            zs_y = []
            for j in range(0,self.h+self.pad,self.stride):
                jl = j #- self.pad
                jh = j + 2*self.pad    
                
                patch = x[...,jl:jh,il:ih]
                u = self.POD(patch)
                z = torch.einsum("BT..., BM... -> BTM", patch, u)
                
                us_y.append(u)
                zs_y.append(z)
            us.append(torch.stack(us_y,dim=-1))
            zs.append(torch.stack(zs_y,dim=-1))
        u = torch.stack(us,dim=-1) # BM C H_patch W_patch y x
        z = torch.stack(zs,dim=-1) # BTM y x
        return u, z
        
    # @torch.compile(fullgraph=False)
    def window_out(self,u,z, x_pad): # same windows??
        x_hat = torch.zeros_like(x_pad,device=self.device)
        norm = torch.zeros((self.h + 2*self.pad, self.w + 2*self.pad),device=self.device)
        
        for x,i in enumerate(range(0,self.w+self.pad,self.stride)):
            il = i #- self.pad
            ih = i + 2*self.pad
            for y,j in enumerate(range(0,self.h+self.pad,self.stride)):
                jl = j #- self.pad
                jh = j + 2*self.pad    
                
                ui = u[...,y,x]
                zi = z[...,y,x]
                # print(ui.shape,zi.shape, jl, jh, il, ih)
                x_hat[...,jl:jh,il:ih] = x_hat[...,jl:jh,il:ih] + torch.einsum("BTM, BM... -> BT...", zi, ui)
                norm[jl:jh,il:ih] = norm[jl:jh,il:ih] + 1.0

        return x_hat / norm
    
    def in_out(self,x_pad):
        u,z = self.window_in(x_pad)
        return self.window_out(u,z,x_pad)

    @torch.compile(fullgraph=False)
    def scalar_gradients_q(self, u, nx, ny):
        # h, w
        # u_xshift = torch.cat([u[...,:,nx:],u[...,:,:nx]],dim=-1)
        # u_yshift = torch.cat([u[...,ny:,:],u[...,:ny,:]],dim=-2)
        # ux = (u_xshift - u) / nx
        # uy = (u_yshift - u) / ny # Meridional derivative
        
        ux_f = torch.cat([u[...,:,nx:] - u[...,:,:-nx],u[...,:,:nx] - u[...,:,-nx:]],dim=-1) / nx
        uy_f = torch.cat([u[...,ny:,:] - u[...,:-ny,:],u[...,:ny,:] - u[...,-ny:,:]],dim=-2) / ny
        
        ux_b = torch.cat([u[...,:,:-nx] - u[...,:,nx:],u[...,:,-nx:] - u[...,:,:nx]],dim=-1) / nx
        uy_b = torch.cat([u[...,:-ny,:] - u[...,ny:,:],u[...,-ny:,:] - u[...,:ny,:]],dim=-2) / ny
        return ux_f, ux_b, uy_f,uy_b
    
    @torch.compile(fullgraph=False)
    def roll(self, u, nx, ny):
        
        ux = torch.cat([u[...,:,nx:],u[...,:,:nx]],dim=-1) # move left side start to nx
        uxy = torch.cat([u[...,ny:,:],u[...,:ny,:]],dim=-2) # move top side start to ny

        return uxy
    
    @torch.compile(fullgraph=False)
    def center(self, u, w):
        # u shape btxy(c)hw,
        # w shape btxyhw
        flat = w.view(*w.shape[:-2],-1)
        flat_index = torch.argmax(flat, dim=-1) # ..., 1 
        index = torch.unravel_index(flat_index, w.shape[-2:]) # ..., 2
        
        uf = torch.empty(u.shape, device=self.device)
        for b in range(u.shape[0]):
            for t in range(u.shape[1]):
                for x in range(u.shape[2]):
                    for y in range(u.shape[3]):
                        u0 = u[b,t,x,y,...]
                        # print(len(index), index[0].shape, u.shape, w.shape)
                        nx = index[0][b,t,x,y]
                        ny = index[1][b,t,x,y]    
                        uf[b,t,x,y,...] = self.roll(u0,nx,ny)    

        return uf, index
    
    @torch.compile(fullgraph=False)
    def uncenter(self, u, index):
        # u shape btn(c)hw,
        uf = torch.empty(u.shape, device=self.device)
        for b in range(u.shape[0]):
            for t in range(u.shape[1]):
                for x in range(u.shape[2]):
                    for y in range(u.shape[3]):
                        u0 = u[b,t,x,y,...]
                        nx = index[0][b,t,x,y]
                        ny = index[1][b,t,x,y]    
                        uf[b,t,x,y,...] = self.roll(u0,-nx,-ny)    

        return uf
    
    def make_patch(self,x_pad):    
        us = []
        for i in range(0,self.w+self.pad,self.stride):
            il = i #- self.pad
            ih = i + 2*self.pad
            us_y = []
            for j in range(0,self.h+self.pad,self.stride):
                jl = j #- self.pad
                jh = j + 2*self.pad    
                
                patch = x_pad[...,jl:jh,il:ih]
                
                us_y.append(patch)
            us.append(torch.stack(us_y,dim=2))
        u = torch.stack(us,dim=3) # BT H_patch W_patch C y x
        return u # windows
    
    def shift_POD(self,windows):
        # assumes windows are B T H_patch W_patch C y x
        # norm = torch.sum(windows**2,dim=-3) # B T H_patch W_patch y x
        # centered_windows, centers = self.center(windows,norm) # B T H_patch W_patch C y x # skipping some times
        s = windows.shape[-2:]
        centered_windows = windows
        # mix patches
        limited_windows = centered_windows[:,:-5,...]
        winds = limited_windows.reshape(centered_windows.shape[0],-1,*centered_windows.shape[4:])
        # print(winds.shape)
        u = self.truncated_POD(winds) # B M C y x
        z = torch.einsum("BMCyx,BTHWCyx->BTHWM",u,centered_windows) # reduced version
        # reproject
        yhat = torch.einsum("BMCyx,BTHWM->BTHWCyx",u,z) # reconstruction
        
        # resolvent is not improving!
        # err = centered_windows - yhat
        # err_fft = torch.fft.rfftn(err,dim=(-2,-1),s=s)
        # limited_err_fft = err_fft[:,:-5,...]
        # err_winds = limited_err_fft.reshape(limited_err_fft.shape[0],-1,*limited_err_fft.shape[4:])
        # u_fft = self.truncated_CPOD(err_winds) # B M C y x
        # z_fft = torch.einsum("BMCyx,BTHWCyx->BTHWM",u_fft,err_fft) #reduced version
        # # reproject
        # err_fft_hat = torch.einsum("BMCyx,BTHWM->BTHWCyx",u_fft,z_fft) # reconstruction
        # err_hat = torch.fft.irfftn(err_fft_hat,dim=(-2,-1),s=s)
        
        # add center
        reco_windows = yhat + err_hat # self.uncenter(yhat, centers) #
        return reco_windows    
        
    def unmake_patch(self,x_pad, windows):    
        x_hat = torch.zeros_like(x_pad,device=self.device)
        norm = torch.zeros((self.h + 2*self.pad, self.w + 2*self.pad),device=self.device)
        
        for x,i in enumerate(range(0,self.w+self.pad,self.stride)):
            il = i #- self.pad
            ih = i + 2*self.pad
            for y,j in enumerate(range(0,self.h+self.pad,self.stride)):
                jl = j #- self.pad
                jh = j + 2*self.pad    
                
                x_hat[...,jl:jh,il:ih] = x_hat[...,jl:jh,il:ih] + windows[...,y,x,:,:,:]
                norm[jl:jh,il:ih] = norm[jl:jh,il:ih] + 1.0

        return x_hat / norm
    
        
    def in_out_POD(self,x_pad):
        windows = self.make_patch(x_pad)
        reco_windows = self.shift_POD(windows)
        return self.unmake_patch(x_pad,reco_windows)
    
    @torch.compile(fullgraph=False)
    def scalar_gradients_raw(self, u):
        ux = torch.gradient(u, dim=-1)[0]  # Zonal derivative
        uy = torch.gradient(u, dim=-2)[0] # Meridional derivative
        return ux, uy
    
    @torch.compile(fullgraph=False)
    def scalar_gradients(self, u, dx, y):
        ux = torch.gradient(u, dim=-1)[0] / dx[None, None, :, None] # Zonal derivative
        uy = torch.gradient(u, spacing=(y,), dim=-2)[0] # Meridional derivative
        uy = torch.clamp(uy,-v_clamp,v_clamp)
        return ux, uy
    
    @torch.compile(fullgraph=False)
    def vector_gradients(self, u, v, dx, y):
        ux = torch.gradient(u, dim=-1)[0] / dx[None, None, :, None] # Zonal derivative # verified
        vx = torch.gradient(v, dim=-1)[0] / dx[None, None, :, None]
        uy = torch.gradient(u, spacing=(y,), dim=-2)[0] # Meridional derivative
        vy = torch.gradient(v, spacing=(y,), dim=-2)[0]
        uy = torch.clamp(uy,-v_clamp,v_clamp)
        vy = torch.clamp(vy,-v_clamp,v_clamp)
        return ux, uy, vx, vy
    
    @torch.compile(fullgraph=False)
    def curl(self, u, v):
        vx = torch.gradient(v, dim=-1)[0]
        uy = torch.gradient(u, dim=-2)[0] 
        return vx-uy
    
    @torch.compile(fullgraph=False)
    def divergence(self, u, v, dx, y):
        ux = torch.gradient(u, dim=-1)[0] / dx[None, None, :, None] # Zonal derivative # can divide afterward since change is perpendicular to gradient direction
        vy = torch.gradient(v, spacing=(y,), dim=-2)[0] # Meridional derivative
        vy = torch.clamp(vy,-v_clamp,v_clamp)
        return ux + vy
    
    @torch.compile(fullgraph=False)
    def _delta_t(self, u):
        return torch.diff(u, dim=1)
    
    @torch.compile(fullgraph=False)
    def _dt(self, u):
        return torch.gradient(u, dim=1)[0]
    
    def calculate_advection(self, u):
        # k = 30
        # kern = torch.ones(u.shape[1],u.shape[1],k,k, device=self.device) / (k**2)
        # u = nn.functional.conv2d(u,kern,padding='same')
        
        ux, uy = self.scalar_gradients_raw(u)
        
        end = -1 # 
        diff_u = self._delta_t(u)
        
        zeros = 0.0 * ux[:,:end]
        ones = zeros + 1e-5 # regularization
        
        A = torch.stack(
            [
                torch.stack(
                    [ux[:,:end], uy[:,:end]], dim = -1 # columns
                ),
                torch.stack(
                    [ones, zeros], dim = -1
                ),
                torch.stack(
                    [zeros, ones], dim = -1
                ),
            ],
            dim = -2 # rows
        ) # rows, cols = n_constraints, 2
        
        b = torch.stack([diff_u, zeros, zeros], dim = -1)[...,None] # rows, cols = n_constraints, 1
        
        # print(A.shape, b.shape)
        
        vel = torch.linalg.lstsq(A,b).solution[...,0] # rows, cols = 2 , 1
        
        return vel[...,0], vel[...,1], ux, uy
    
    def calculate_advection_pyramid(self, u):
        
        # R is leftover from diff_u = self._delta_t(u)
        R = self._delta_t(u)
        res = 2
        
        assert self.h % res == 0
        assert self.w % res ==0      
        resmax = int(math.log2(min(self.h,self.w)))
        
        rx = 2
        ry = 2
        
        # pool = torch.nn.AvgPool3d((ry,rx)) # lambda x : x 
        # upsample = lambda u: torch.repeat_interleave(torch.repeat_interleave(u, rx, dim=-2), ry, dim=-3) #torch.nn.Upsample(size=(self.h,self.w, 4)) # lambda x: x 
        # pool = torch.nn.AvgPool2d((ry,rx))
        upsample = torch.nn.Upsample(size=(self.h,self.w))
        
        end = -1 # 
        with torch.no_grad():
        # zeros = 0.0 * ux[:,:end]
        # ones = zeros + 1e-5 # regularization
        # modes = []
            cs = []
            recos = []
            recos.append(R)
            u_past = 0
            for res_2 in range(1,resmax):
                res = 2**res_2
                # print(f"RES: {res}")
                nx = self.w // res
                ny = self.h // res  
                
                
                # u_hf = u - u_past # higher frequencies
                # if ny % 2 == 0:
                #     ny += 1
                # if nx % 2 == 0:
                #     nx += 1
                
                # gblur = tf.GaussianBlur((ny,nx), sigma=0.1)
                # u_pool = gblur(u_hf) # only get next-frequency information
                # R_pool = gblur(R)
                
                ux_f,ux_b,uy_f,uy_b = self.scalar_gradients_q(u_pool,1,1) #self.scalar_gradients_q(torch.nn.functional.avg_pool2d(u,(nx,ny)),1,1)#self.scalar_gradients_q(u,nx,ny)
                
                c = self.lstsq_reduction(ux_f[:,:end], ux_b[:,:end], uy_f[:,:end], uy_b[:,:end], R_pool) # BTM # torch.nn.functional.avg_pool2d(R,(nx,ny))
                rc = (self.lstsq_projection(ux_f[:,:end], ux_b[:,:end], uy_f[:,:end], uy_b[:,:end], c))
                R = R - rc
                u_past = u_past + (u_pool) # this frequency is over
        
                # modes.append(m)
                cs.append(c)
                recos.append(R)
            
        recos = torch.stack(recos,dim=2)
        recos = torch.cumsum(recos,dim=2) # BTMHW
        return recos
    
    def lstsq_reduction(self,ux_f,ux_b,uy_f,uy_b,du):
        # downsample
        # ux_f = pool(ux_f)
        # uy_f = pool(uy_f)
        # ux_b = pool(ux_b)
        # uy_b = pool(uy_b)
        # du = pool(du)        
        
        zeros = 0.0 * ux_f
        ones = zeros + 1e-5 # regularization
        
        uxy = torch.stack([ux_f, ux_b, uy_f, uy_b], dim = -1) # columns
                
        A = torch.stack(
            [
                uxy,
                torch.stack(
                    [ones, zeros, zeros, zeros], dim = -1
                ),
                torch.stack(
                    [zeros, ones, zeros, zeros], dim = -1
                ),
                torch.stack(
                    [zeros, zeros, ones, zeros], dim = -1
                ),
                torch.stack(
                    [zeros, zeros, zeros, ones], dim = -1
                ),
            ],
            dim = -2 # rows
        ) # rows, cols = n_constraints, 2
        
        b = torch.stack([du, zeros, zeros, zeros, zeros], dim = -1)[...,None] # rows, cols = n_constraints, 1

        return torch.linalg.lstsq(A,b).solution[...,0] # rows, cols = 2 , 1
    
    def lstsq_projection(self,ux_f,ux_b,uy_f,uy_b,c):
        uxy = torch.stack([ux_f, ux_b, uy_f, uy_b], dim = -1) # columns
        # cu = upsample(c)
        # print(c.shape, cu.shape, uxy.shape)
        return torch.sum(uxy * c,dim=-1) # BTHW
    
    def advect_step(self,vx,vy,ux,uy,u):
        return u + ux * vx + uy * vy
          
    def calculate_advection_2(self, u):
        # pretend we have a basis in the spatial gradients?? of course, this fails..
        
        ux, uy = self.scalar_gradients_raw(u) # BTHW
        
        ux_f = ux.reshape(*(ux.shape[:2]),-1) # BT(HW)
        uy_f = uy.reshape(*(uy.shape[:2]),-1) # BT(HW)
        
         
        diff_u = self._delta_t(u) # B(T-1)HW
        du_t_f = diff_u.reshape(*(diff_u.shape[:2]),-1) # B(HW)(T-1)
        
        hist = 8
        nb = u.shape[0]
        
        A_list = [torch.cat([ux_f[:,i-hist:i],uy_f[:,i-hist:i]],dim=1).permute(0,2,1) for i in range(hist, ux.shape[1])] # B HW, 2*hist

        A = torch.stack(A_list,dim = 1) # B, windows, HW, time along window
        b = du_t_f[:, hist-1:ux.shape[1]-1][...,None] # B, windows, HW, 1
        
        print(A.shape, b.shape)
        # A2 = torch.stack([*A_list,*A_list2],dim=-2) # all windows, time along window
        
        # print(A.shape, b.shape, A2.shape)
        
        cxy = torch.linalg.lstsq(A,b).solution #  B, windows, time along window, 1
        
        reco = (A @ cxy)[0].reshape(u.shape[0], -1, *u.shape[2:]) + u[:,hist:]
        err = u[:,hist+1:]-reco[:,:-1]
        
        return reco[:,:-1], err

    def mode_adv(self, u):
        # btchw
        hist = 10
        patches = [u[:,i-hist:i] for i in range(hist, u.shape[1])]
        u_for_reco = torch.stack([p[:,-1] for p in patches], dim=1)
        modes = [self.POD(ui) for ui in patches]
        
        coeffs = torch.stack([torch.einsum("B...,BM...->BM",x[:,-1,...],u_pod) for u_pod, x in zip(modes, patches)],dim=1) # B T_which_patch M # get last coefficient set
        
        modes_ = [torch.stack([p[:,i,...] for p in modes], dim = 1) for i in range(hist)]
        
        modes_shift = []
        vx_list = []
        vy_list = []
        for i,m in enumerate(modes_):
            vx,vy,ux,uy = self.calculate_advection(m)
            m_shift = self.advect_step(vx,vy,ux[:,:-1],uy[:,:-1],m[:,:-1]) # T-1 since prediction did not happen
            modes_shift.append(m_shift)
            vx_list.append(vx)
            vy_list.append(vy)
            # step along T_which_patch
        
        shifted_modes = torch.stack(modes_shift, dim=1) # B, M, T_which_patch, ...
        shifted_reco = torch.einsum("BMT...,BTM->BT...",shifted_modes,coeffs[:,1:]) # shift coeffs as well, and only T-1 since prediction did not happen.

        err = u_for_reco[:,1:] - shifted_reco
        vx = torch.stack(vx_list, dim=1)
        vy = torch.stack(vy_list, dim=1)
        return shifted_reco, err, shifted_modes, u_for_reco[:,1:], vx, vy      

    def fft_step(self,u):
        u_fft = torch.fft.rfft(u, dim=1, n = u.shape[1]) # time-based FFT
        w = (torch.fft.rfftfreq(u.shape[1],device=self.device) * 2 * pi)[None,:,None,None,None].expand(u_fft.shape)
        iw = torch.complex(0*w, w) # real, imag
        du_fft = iw * u_fft
        
        du = torch.fft.irfft(du_fft, dim=1, n = u.shape[1])
        return u + du 
    
    def lstsq(self,u,m):
        # BT..., BMTchw, WANT BTM
        m2 = m.permute(0,2,3,4,5,1).reshape(m.shape[0],m.shape[2],-1,m.shape[1]) # BTchwM
        u2 = u.reshape(u.shape[0],u.shape[1],-1)[...,None]
        c = torch.linalg.lstsq(m2,u2).solution[...,0]
        return c
    
    def lstsq_no_c(self,u,m):
        # BT..., BMThw, WANT BTM
        m2 = m.permute(0,2,3,4,1).reshape(m.shape[0],m.shape[2],-1,m.shape[1]) # BThwM
        u2 = u.reshape(u.shape[0],u.shape[1],-1)[...,None]
        c = torch.linalg.lstsq(m2,u2).solution[...,0]
        return c
    
    def mode_adv_2(self, u):
        # btchw
        hist = 10
        patches = [u[:,i-hist:i] for i in range(hist, u.shape[1])]
        u_for_reco = torch.stack([p[:,-1] for p in patches], dim=1)
        modes = [self.POD(ui) for ui in patches]
        
        coeffs = torch.stack([torch.einsum("B...,BM...->BM",x[:,-1,...],u_pod) for u_pod, x in zip(modes, patches)],dim=1) # B T_which_patch M # get last coefficient set
        
        modes_ = [torch.stack([p[:,i,...] for p in modes], dim = 1) for i in range(hist)]
        
        modes_shift = []
        vx_list = []
        vy_list = []
        for i,m in enumerate(modes_):
            m_shift = self.fft_step(m)
            modes_shift.append(m_shift)
            # step along T_which_patch
        
        shifted_modes = torch.stack(modes_shift, dim=1) # B, M, T_which_patch, ...
        
        coeffs2 = self.lstsq(u_for_reco[:,1:],shifted_modes[:,:,:-1])
        
        shifted_reco = torch.einsum("BMT...,BTM->BT...",shifted_modes[:,:,:-1],coeffs2) # shift coeffs as well, and only T-1 since prediction did not happen.

        err = u_for_reco[:,1:] - shifted_reco
        return shifted_reco, err, shifted_modes[:,:,:-1], u_for_reco[:,1:]   
    
    def calculate_terms(self, u, v, u0, v0, p, dx, y):
        # BTHW
        with torch.no_grad():
            ux, uy, vx, vy = self.vector_gradients(u, v, dx, y)

            # approximation for time-step (which we want to learn)
            ut = torch.diff(u, dim=1) / self.dt
            vt = torch.diff(v, dim=1) / self.dt
            
            # # Coriolis forces
            # f = self.f_star
            # cor_u = torch.einsum('h,bthw->bthw',f,v)
            # cor_v = -torch.einsum('h,bthw->bthw',f,u)
            
            # pressure terms
            px, py = self.scalar_gradients(p, dx, y)
            
            # advection terms (incl.coriolis forces)
            advx = u0*ux + v0*uy #+ cor_u 
            advy = u0*vx + v0*vy #+ cor_v 
            
            # source terms
            fx = (ut + advx[:,:-1] + px[:,:-1])
            fy = (vt + advy[:,:-1] + py[:,:-1])
            
            # so we can recalculate ut, vt using the fx, fy values as opposed to direct estimation (which is lossier / more lossy)
            # ut = fx - advx - px
            # vt = fy - advy - py
        
        return fx, fy, advx, advy, px, py
    
    # @torch.compile(fullgraph=False)
    def window_in(self,x):    
        us = []
        zs = []
        for i in range(0,self.w+self.pad,self.stride):
            il = i #- self.pad
            ih = i + 2*self.pad
            us_y = []
            zs_y = []
            for j in range(0,self.h+self.pad,self.stride):
                jl = j #- self.pad
                jh = j + 2*self.pad    
                
                patch = x[...,jl:jh,il:ih]
                u = self.POD(patch)
                z = torch.einsum("BT..., BM... -> BTM", patch, u)
                
                us_y.append(u)
                zs_y.append(z)
            us.append(torch.stack(us_y,dim=-1))
            zs.append(torch.stack(zs_y,dim=-1))
        u = torch.stack(us,dim=-1) # BM C H_patch W_patch y x
        z = torch.stack(zs,dim=-1) # BTM y x
        return u, z

    def calculate_step(self, u, v, u0, v0, p, dx, y, pos):
        fx,fy, advx, advy, px, py = self.calculate_terms(u, v, u0, v0, p, dx, y) # all past pressures: B (T-1) HW, and all adv terms:  BTHW
        ufx = torch.mean(fx,dim=1,keepdim=True)
        ufy = torch.mean(fy,dim=1,keepdim=True)
        fx = fx - ufx
        fy = fy - ufy
        
        fxy = torch.stack([fx,fy],dim=2) # BTCHW
        # fxy_pad = self.pad_in(fxy)
        # shifted_fxy = self.apply_CNN_explicit_single_step(fxy_pad, fxy.shape, pos)
        
        shifted_fxy = self.apply_CNN_explicit(fxy, pos)
        # print("shifted done")
        
        shift_fx = shifted_fxy[:,:,0,...]
        shift_fy = shifted_fxy[:,:,1,...]
        
        with torch.no_grad():
            import matplotlib.pyplot as plt
            # print(dx)
            # torch.gradient(u, dim=-1)[0]
            # dx[None, None, :, None].expand(u.shape)
            # y[None, None, :, None].expand(u.shape)
            self.iter += 1
            if self.iter % self.print_every == 0:
                for i in range(0,fx.shape[1]-1,5):
                    plt.imshow((shift_fx[0,i,:,:] - fx[0,i+1,:,:]).cpu().numpy())
                    plt.colorbar()
                    plt.clim(-60,60)
                    plt.savefig(f"fx_{self.iter}_{i}.png")
                    plt.close()
                for i in range(0,fy.shape[1]-1,5):
                    plt.imshow((shift_fy[0,i,:,:] - fy[0,i+1,:,:]).cpu().numpy())
                    plt.colorbar()
                    plt.clim(-60,60)
                    plt.savefig(f"fy_{self.iter}_{i}.png")
                    plt.close()
            
            # 
            # vx, vy, ux, uy = self.calculate_advection(fx)
            # vxx, vxy, _,_ = self.calculate_advection(vx)
            # fx_reco = self.advect_step(vx,vy,ux[:,:-1],uy[:,:-1],fx[:,:-1])
            # # c = vx
            # # fxy_u = self.POD(fx)
            # # z = torch.einsum("BT..., BM... -> BTM", fx, fxy_u)
            # # c = torch.einsum("BTM, BM... -> BT...", z, fxy_u)
            # # fxy_pad = self.pad_in(fxy)
            # # fxy2 = self.window_in_out_2(fxy_pad)
            # # c = fxy2[:,:,0]
            # # fx_pad = fxy_pad[:,:,0]
            # # c = self.curl(fx,fy)
            # uvx = torch.mean(vx, dim = 1, keepdim=True)
            # vx_dm = vx - uvx
            
            # uvxx = torch.mean(vxx, dim=1, keepdim=True)
            # vxx_dm = vxx-uvxx
            
            # for i in range(vx.shape[1]-1):
            #     fig, axs = plt.subplots(2,1)
            #     q = axs[0].imshow(vx[0,i,:,:].cpu().numpy())
            #     plt.colorbar(q)
            #     q.set_clim(-60,60)
            #     # q = axs[1].imshow((fx_reco[0,i,:,:] - fx[0,i+1,:,:]).cpu().numpy())
            #     q = axs[1].imshow((vy[0,i,:,:]).cpu().numpy())
            #     plt.colorbar(q)
            #     q.set_clim(-60,60)
            #     plt.savefig(f"vxy_{i}.png")
            #     plt.close()

            # fx_shift, err = self.calculate_advection_2(fx)
            
            # m = self.calculate_advection_pyramid(fx)
            # for i in range(m.shape[2]):
            #     q = plt.imshow(m[0,2,i,...].cpu().numpy()) # B,T,M,H,W: fx modes on time 2
            #     plt.colorbar(q)
            #     plt.title(f"2nd_batch_mode_component_{i}")
            #     # plt.clim(-100,100)
            #     plt.savefig(f"fx_{i}.png")
            #     plt.close()
                

            # reco, err, m, gt = self.mode_adv_2(fxy)
            # for i in range(m.shape[2]):
            #     q = plt.imshow(m[0,i,2,0,...].cpu().numpy()) # B,M,T,C,H,W: fx modes on time 2
            #     plt.colorbar(q)
            #     plt.title(f"2nd_batch_mode_{i}")
            #     plt.savefig(f"m_{i}.png")
            #     plt.close()
                
            # for i in range(vx.shape[2]):
            #     q = plt.imshow(vx[0,i,2,0,...].cpu().numpy()) # B,M,T,C,H,W: fx modes on time 2
            #     plt.colorbar(q)
            #     plt.title(f"2nd_batch_vx_{i}")
            #     plt.savefig(f"vx_{i}.png")
            #     plt.close()
                
            # for i in range(vy.shape[2]):
            #     q = plt.imshow(vy[0,i,2,0,...].cpu().numpy()) # B,M,T,C,H,W: fx modes on time 2
            #     plt.colorbar(q)
            #     plt.title(f"2nd_batch_vy_{i}")
            #     plt.savefig(f"vy_{i}.png")
            #     plt.close()   
            
            # fxy = torch.stack([fx,fy],dim=2) # BTCHW
            # gt = self.pad_in(fxy)
            # reco = self.in_out_POD(gt)
            # err = gt-reco
            # for i in range(reco.shape[1]):
            #     fig, axs = plt.subplots(3,1,figsize=(10,10))
            #     q = axs[0].imshow(gt[0,i,0,:,:].cpu().numpy()) # BTCHW
            #     plt.colorbar(q)
            #     q.set_clim(-60,60)
            #     axs[0].set_title("ground_truth")
            #     # q = axs[1].imshow((fx_reco[0,i,:,:] - fx[0,i+1,:,:]).cpu().numpy())
            #     q = axs[1].imshow(reco[0,i,0,:,:].cpu().numpy())
            #     plt.colorbar(q)
            #     q.set_clim(-60,60)
            #     axs[1].set_title("reco")
            #     q = axs[2].imshow(err[0,i,0,:,:].cpu().numpy())
            #     plt.colorbar(q)
            #     # q.set_clim(-60,60)
            #     axs[2].set_title("err")
            #     plt.tight_layout()
            #     plt.savefig(f"reco_fx_{i}.png")
            #     plt.close()
            
            # quit()
        
        fx_shift = shift_fx[:,-1:] + ufx #self.source_DMD(fx)
        fy_shift = shift_fy[:,-1:] + ufy #self.source_DMD(fy) # get shifted pressures, to get the last-step pressure.
                
        advx_shift = advx[:,-1:] + px[:,-1:]
        advy_shift = advy[:,-1:] + py[:,-1:]
         
        ut = fx_shift - advx_shift
        vt = fy_shift - advy_shift # B_HW,
        
        u2 = u[:,-1:] + ut * self.dt # BTHW (though T=1)
        v2 = v[:,-1:] + vt * self.dt
            
        return u2, v2 # last timesteps


    def concentration_resnet(self, c, u, v, dx, y):
        # BCHW with the exception of the two velocities
        # TODO add source term
        uc = (self.rate[None,:,:,:,0] * u[:,None,:,:]) * c
        vc = (self.rate[None,:,:,:,1] * v[:,None,:,:]) * c
        
        with torch.no_grad():
            cx, cy = self.scalar_gradients(c, dx, y)
        
        innerx = cx * self.diffusivity[None,:,None,None] - uc
        innery = cy * self.diffusivity[None,:,None,None] - vc
        
        dcdt = self.divergence(innerx, innery, dx, y)
        # assert torch.all(torch.isfinite(dcdt))
        c2 = c + dcdt * self.dt * 1e-12
        # assert torch.all(torch.isfinite(c2))
        # # add cohesion loss
        if self.training:
            loss = self.c_cohese * torch.nn.functional.mse_loss(c2[:,:-1],c[:,1:])
            loss.backward(retain_graph=True) # only if training!
        
        return c2[:,-1:]
    
    def forward(self, x):
        assert x.shape[1] > 2, "Not enough history. Coherency losses will fail"
        
        # make grid
        dx = self.dx(self.l_star)
        y = self.y(self.l_star)
        two_wx = self.wx(self.l_star) # along latitude (y)
        pos_encoding = self.pos(self.l_star)
        # wwx = self.wwx(self.l_star)
        # BTCHW, c==0 => u, c==1 => v
        u0 = x[:,:,0,:,:] 
        v0 = x[:,:,1,:,:]
        u = u0 + two_wx[None,None,:,None]
        v = v0 
        p = x[:,:,3,:,:] * const_P
        
        c = x[:,:,2:,:,:] # only temp and pressure for now
        
       
        u2,v2 = self.calculate_step(u, v, u0, v0, p, dx, y, pos_encoding) # bad for some reason
        # c2 = self.concentration_resnet(c,u[:,-1],v[:,-1], dx, y)
        
        out = torch.cat([u2[:,:,None,:,:],v2[:,:,None,:,:],c[:,-1:,...]],dim=2)  
        return out

    
    
    