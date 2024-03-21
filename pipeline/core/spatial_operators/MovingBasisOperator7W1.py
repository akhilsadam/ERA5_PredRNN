__author__ = 'asadam'

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_harmonics as th
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
        
        self.POD_kern = 4 # even
        self.pad = (self.POD_kern)//2
        self.stride = self.pad
        
        # local (neighbor modes)
        self.neighbor_kern = 8 # this times stride must be less than padding! NOTE: add padding increase eventually to accomodate large stride..
        self.neighbor_stride = 1
        
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
    def pad_in(self,x):
        shape = x.shape[:-2]        
        x = x.reshape((x.shape[0]*x.shape[1]*x.shape[2], self.h, self.w))
        x = F.pad(x, (self.pad,self.pad), mode='circular')
        x = F.pad(x, (0,0,self.pad,self.pad), mode='reflect') # size is (BxTxC)hw, where hw = H+2p, W+2p
        x = x.reshape((*shape, self.h + 2*self.pad, self.w + 2*self.pad))
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
        ones = zeros + 0.1 # regularization
        
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
        modes = [self.POD(ui) for ui in patches]
        
        coeffs = [torch.einsum("BT...,BM...->BTM",x,u_pod) for u_pod, x in zip(modes, patches)]
        
        
        

    def advect_step(self,vx,vy,ux,uy,u):
        return u + ux * vx + uy * vy
          
    def calculate_terms(self, u, v, p, dx, y):
        # BTHW
        with torch.no_grad():
            ux, uy, vx, vy = self.vector_gradients(u, v, dx, y)

            # approximation for time-step (which we want to learn)
            ut = torch.diff(u, dim=1) / self.dt
            vt = torch.diff(v, dim=1) / self.dt
            
            # Coriolis forces
            f = self.f_star
            cor_u = torch.einsum('h,bthw->bthw',f,v)
            cor_v = -torch.einsum('h,bthw->bthw',f,u)
            
            # pressure terms
            px, py = self.scalar_gradients(p, dx, y)
            
            # advection terms (incl.coriolis forces)
            advx = u*ux + v*uy + cor_u 
            advy = u*vx + v*vy + cor_v 
            
            # source terms
            fx = (ut + advx[:,:-1] + px[:,:-1])
            fy = (vt + advy[:,:-1] + py[:,:-1])
            
            # so we can recalculate ut, vt using the fx, fy values as opposed to direct estimation (which is lossier / more lossy)
            # ut = fx - advx - px
            # vt = fy - advy - py
        
        return fx, fy, advx, advy, px, py
    

    def calculate_step(self, u, v, p, dx, y):
        fx,fy, advx, advy, px, py = self.calculate_terms(u, v, p, dx, y) # all past pressures: B (T-1) HW, and all adv terms:  BTHW
        ufx = torch.mean(fx,dim=1,keepdim=True)
        ufy = torch.mean(fy,dim=1,keepdim=True)
        fx = fx - ufx
        fy = fy - ufy
        
        with torch.no_grad():
            import matplotlib.pyplot as plt
            # print(dx)
            # torch.gradient(u, dim=-1)[0]
            # dx[None, None, :, None].expand(u.shape)
            # y[None, None, :, None].expand(u.shape)
            # for i in range(fx.shape[1]):
            #     plt.imshow(fx[0,i,:,:].cpu().numpy())
            #     plt.colorbar()
            #     plt.clim(-60,60)
            #     plt.savefig(f"fx_{i}.png")
            #     plt.close()
            # for i in range(fy.shape[1]):
            #     plt.imshow(fy[0,i,:,:].cpu().numpy())
            #     plt.colorbar()
            #     plt.clim(-150,150)
            #     plt.savefig(f"fy_{i}.png")
            #     plt.close()
            
            fxy = torch.stack([fx,fy],dim=2) # BTCHW
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
            #     q = axs[0].imshow(vx_dm[0,i,:,:].cpu().numpy())
            #     plt.colorbar(q)
            #     q.set_clim(-60,60)
            #     # q = axs[1].imshow((fx_reco[0,i,:,:] - fx[0,i+1,:,:]).cpu().numpy())
            #     q = axs[1].imshow((vxx_dm[0,i+1,:,:]).cpu().numpy())
            #     plt.colorbar(q)
            #     q.set_clim(-60,60)
            #     plt.savefig(f"reco_fx_{i}.png")
            #     plt.close()

            # fx_shift, err = self.calculate_advection_2(fx)

            m, c, reco = self.mode_adv(fxy)
             
            for i in range(fx_shift.shape[1]-1):
                fig, axs = plt.subplots(2,1)
                q = axs[0].imshow(m[0,i,0,:,:].cpu().numpy())
                plt.colorbar(q)
                q.set_clim(-60,60)
                # q = axs[1].imshow((fx_reco[0,i,:,:] - fx[0,i+1,:,:]).cpu().numpy())
                q = axs[1].imshow(reco[0,i,0,:,:].cpu().numpy())
                plt.colorbar(q)
                q.set_clim(-60,60)
                plt.savefig(f"reco_fx_{i}.png")
                plt.close()
            
            # quit()
        
        fx_shift = fx[:,-1:] + ufx #self.source_DMD(fx)
        fy_shift = fy[:,-1:] + ufy #self.source_DMD(fy) # get shifted pressures, to get the last-step pressure.
                
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
        
        # BTCHW, c==0 => u, c==1 => v
        u = x[:,:,0,:,:]
        v = x[:,:,1,:,:]
        p = x[:,:,3,:,:] * const_P
        
        c = x[:,:,2:,:,:] # only temp and pressure for now
        
        c2 = self.concentration_resnet(c,u[:,-1],v[:,-1], dx, y)
        u2,v2 = self.calculate_step(u, v, p, dx, y) # bad for some reason

        out = torch.cat([u2[:,:,None,:,:],v2[:,:,None,:,:],c2],dim=2)  
        return out

    
    
    