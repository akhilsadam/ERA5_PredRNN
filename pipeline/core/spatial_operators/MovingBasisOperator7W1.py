__author__ = 'asadam'

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_harmonics as th
import math

from pipeline.normalize import norm_scales as scale

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


# 2D incompressible N-S

def _matrix_pow(matrix: torch.Tensor, p: float) -> torch.Tensor:
    r"""
    Power of a matrix using Eigen Decomposition.
    Args:
        matrix: matrix
        p: power
    Returns:
        Power of a matrix
    """
    vals, vecs = torch.eig(matrix, eigenvectors=True)
    vals = torch.view_as_complex(vals.contiguous())
    vals_pow = vals.pow(p)
    vals_pow = torch.view_as_real(vals_pow)[..., 0]
    matrix_pow = torch.matmul(vecs, torch.matmul(torch.diag(vals_pow), torch.inverse(vecs)))
    return matrix_pow

class Operator(nn.Module):
    def __init__(self, nlatent, ntime, h, w, device, n_embd=400, nlayers=1, activation=torch.nn.ReLU(), **kwargs):
        super(Operator, self).__init__()
        
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
        self.lat = torch.linspace(0,2*pi,steps=h, device=device)

        # Step size as a function of latitude (in m)        
        self.l_star = torch.tensor([l_star],device=device)
        self.dx = lambda r: (1/h)*2*pi*r*torch.cos(self.lat)# Radius as a function of latitude
        self.y = lambda dx: torch.cumsum(dx, dim=0)
        self.dt = torch.tensor([dt_star],device=device) # 6-Hourly resolution # TODO make parameter

        # Coriolis parameter
        self.f_star = 2*omega_star*torch.sin(self.lat) # Rotation rate (coriolis, 2w from 2w x v)
        
        
        
        # self.rate = nn.Parameter(torch.ones((self.clatent,h,w,2),device=device)) # friction-modulated velocity adjustment
        # self.diffusivity = nn.Parameter(torch.zeros((self.clatent),device=device)) # TODO make space-dependent!
        
        n_modes = ntime-1
        self.Ax = nn.Parameter(torch.empty((n_modes,n_modes),device=device))
        nn.init.xavier_uniform_(self.Ax.data,gain=0.001)
        self.Ay = nn.Parameter(torch.empty((n_modes,n_modes),device=device))
        nn.init.xavier_uniform_(self.Ay.data,gain=0.001)
        
        # regularization
        # self.f_cohese = 1e-4
        # self.v_cohese = 1e-2
        self.c_cohese = 1e-2
        
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
    def source_DMD(self,x):
        with torch.no_grad():
            u = self.POD(x) # BM...
            z = torch.einsum("BT..., BM... -> BTM", x, u)
            
            D_plus = torch.linalg.lstsq(z[:,:-1,:],z[:,1:,:]) # B,t,m_in  + B,t,m_out -> B,m_in,m_out           # forward DMD (AX-B)
            D_minus = torch.linalg.lstsq(z[:,1:,:],z[:,:-1,:]) #  -> B,m_out,m_in (since reverse direction)     # backward DMD
            D_plusminus = _matrix_pow(torch.einsum("Bba,Bbc->Bca",D_plus,D_minus),0.5)                          # square root of product
            
            
            z_step = torch.einsum("Bc,Bca->Ba",z[:,-1,:],D_plusminus)           # step
            x_new = torch.einsum("Bm, Bm... -> B...", z_step, u)[:,None,...]    # reproject
            # y = torch.cat([x,x_new],dim=1) # B(T+1)...                          # cat
            
        return x_new
    
    @torch.compile(fullgraph=False)
    def scalar_gradients(self, u, dx, y):
        ux = torch.gradient(u, dim=-1)[0] / dx[None, None, :, None] # Zonal derivative
        uy = torch.gradient(u, spacing=(y,), dim=-2)[0] # Meridional derivative
        return ux, uy
    
    @torch.compile(fullgraph=False)
    def vector_gradients(self, u, v, dx, y):
        ux = torch.gradient(u, dim=-1)[0] / dx[None, None, :, None] # Zonal derivative
        vx = torch.gradient(v, dim=-1)[0] / dx[None, None, :, None]
        uy = torch.gradient(u, spacing=(y,), dim=-2)[0] # Meridional derivative
        vy = torch.gradient(v, spacing=(y,), dim=-2)[0]
        return ux, uy, vx, vy

    @torch.compile(fullgraph=False)
    def divergence(self, u, v, dx, y):
        ux = torch.gradient(u, dim=-1)[0] / dx[None, None, :, None] # Zonal derivative # can divide afterward since change is perpendicular to gradient direction
        vy = torch.gradient(v, spacing=(y,), dim=-2)[0] # Meridional derivative
        return ux + vy
          
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
        
        fx_shift = self.source_DMD(fx)
        fy_shift = self.source_DMD(fy) # get shifted pressures, to get the last-step pressure.
                
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
        assert torch.all(torch.isfinite(dcdt))
        c2 = c + dcdt * self.dt   
        assert torch.all(torch.isfinite(c2))
        # # add cohesion loss
        if self.training:
            loss = self.c_cohese * torch.nn.functional.mse_loss(c2[:,:-1],c[:,1:])
            loss.backward(retain_graph=True) # only if training!
        
        return c2[:,-1:]
    
    def forward(self, x):
        assert x.shape[1] > 2, "Not enough history. Coherency losses will fail"
        
        # make grid
        dx = self.dx(self.l_star)
        y = self.y(dx)
        
        # BTCHW, c==0 => u, c==1 => v
        u = x[:,:,0,:,:]
        v = x[:,:,1,:,:]
        p = x[:,:,3,:,:] * const_P
        
        c = x[:,:,2:,:,:] # only temp and pressure for now
        
        c2 = self.concentration_resnet(c,u[:,-1],v[:,-1], dx, y)
        u2,v2 = self.calculate_step(u, v, p, dx, y) # bad for some reason

        y = torch.cat([u2[:,:,None,:,:],v2[:,:,None,:,:],c2],dim=2)  
        return y

    
    
    