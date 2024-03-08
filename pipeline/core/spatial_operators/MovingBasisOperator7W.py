__author__ = 'asadam'

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_harmonics as th
import math




# 2D incompressible N-S



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
        
        pi = math.pi   

        lat = torch.linspace(0,360,steps=h, device=device)
        self.nlat = h
        self.nlon = w
        
        dt_scale = 6*3600.0
        
        self.lat_rad = pi*lat/180
        self.R_earth = 6370 # Earth's radius (km)
        self.R_lat = self.R_earth*torch.cos(self.lat_rad)  # Radius as a function of latitude

        # Step size as a function of latitude (in m)
        #   Based on 1/25 degree resolution
        self.dx = (1/h)*2*pi*self.R_lat*1000 / dt_scale
        self.y = torch.cumsum(self.dx, dim=0)
        self.dt = torch.tensor([1],device=device) # 6-Hourly resolution # TODO make parameter

        # Coriolis parameter
        self.omega = 7.3e-5 # Rotation rate
        self.f = 2*self.omega*torch.sin(self.lat_rad) # s^-1 # size h

        # Density
        # self.rho = nn.Parameter([1.204],device=device) # kg/m^3 # assuming incompressible for now! NEED TO CHANGE
        
        self.rate = nn.Parameter(torch.ones((self.clatent,h,w,2),device=device)) # friction-modulated velocity adjustment
        
        self.diffusivity = nn.Parameter(torch.zeros((self.clatent),device=device)) # TODO make space-dependent!
        
        n_modes = ntime-1
        self.Ax = nn.Parameter(torch.empty((n_modes,n_modes),device=device))
        nn.init.xavier_uniform_(self.Ax.data,gain=0.1)
        self.Ay = nn.Parameter(torch.empty((n_modes,n_modes),device=device))
        nn.init.xavier_uniform_(self.Ay.data,gain=0.1)
        
        # regularization
        self.f_cohese = 1e-4
        self.v_cohese = 1e-2
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
    
    def step(self,x,A):
        # BTCHW
        u = self.POD(x) # BMHW
        y = torch.einsum("BTHW, BMHW, Mm, Bmhw -> BThw",x,u,A,u) # single shift
        return y
    
    @torch.compile(fullgraph=False)
    def scalar_gradients(self,u):
        # Zonal derivative
        ux = torch.zeros(u.shape,device=self.device)
        for i in range(self.nlat):
            ux[..., i, :] = torch.gradient(u[..., i, :], spacing=self.dx[i].item(), dim=-1)[0]
            
        # Meridional derivative
        uy = torch.zeros(u.shape,device=self.device)
        for i in range(self.nlon):
            uy[..., :, i] = torch.gradient(u[..., :, i], spacing=(self.y,), dim=-1)[0]
        
        return ux, uy
    
    @torch.compile(fullgraph=False)
    def vector_gradients(self,u,v):
        # Zonal derivative
        ux = torch.zeros(u.shape,device=self.device)
        vx = torch.zeros(u.shape,device=self.device)
        for i in range(self.nlat):
            ux[..., i, :] = torch.gradient(u[..., i, :], spacing=self.dx[i].item(), dim=-1)[0]
            vx[..., i, :] = torch.gradient(v[..., i, :], spacing=self.dx[i].item(), dim=-1)[0]
            
        # Meridional derivative
        uy = torch.zeros(u.shape,device=self.device)
        vy = torch.zeros(u.shape,device=self.device)
        for i in range(self.nlon):
            uy[..., :, i] = torch.gradient(u[..., :, i], spacing=(self.y,), dim=-1)[0]
            vy[..., :, i] = torch.gradient(v[..., :, i], spacing=(self.y,), dim=-1)[0]
        
        return ux, uy, vx, vy

    # @torch.compile(fullgraph=False)
    def divergence(self,u,v):
        # Zonal derivative
        # ux = torch.zeros(u.shape,device=self.device,requires_grad=True)
        # for i in range(self.nlat):
        #     ux[..., i, :] = torch.gradient(u[..., i, :], spacing=self.dx[i].item(), dim=-1)[0]
        ux = torch.stack([torch.gradient(u[..., i, :], spacing=self.dx[i].item(), dim=-1)[0] for i in range(self.nlat)], dim=-2)
            
        # Meridional derivative
        # vy = torch.zeros(u.shape,device=self.device,requires_grad=True)
        # for i in range(self.nlon):
        #     vy[..., :, i] = torch.gradient(v[..., :, i], spacing=(self.y,), dim=-1)[0]
        vy = torch.stack([torch.gradient(v[..., :, i], spacing=(self.y,), dim=-1)[0] for i in range(self.nlon)], dim=-1)
        return ux + vy
    
    # NOTE obsoleted, may use later
    # def t_dot_err(self,ut,vt,u,v,fx,fy):
    #     # get residual to solve for ut, vt by approximate fixed point.
    #     u2 = ut * self.dt + u
    #     v2 = vt * self.dt + v 
               
    #     ux, uy, vx, vy = self.vector_gradients(u2,v2)

    #     # Material derivatives
    #     Du = ut + u2*ux + v2*uy;
    #     Dv = vt + u2*vx + v2*vy;

    #     # Coriolis forces
    #     cor_u = torch.einsum('h,bthw->bthw',self.f,v2)
    #     cor_v = -torch.einsum('h,bthw->bthw',self.f,u2)
        
    #     # unknown negative pressure gradients
    #     fx_hat = (Du + cor_u) # divided by density, negative (since this is just the residual directly)
    #     fy_hat = (Dv + cor_v)
        
    #     errut = fx_hat - fx
    #     errvt = fy_hat - fy # these are same as ut,vt errors by linearity (here we approximate the advection and coriolis terms relatively constant)
        
    #     return errut, errvt, u2, v2

    # time derivative
    # ut = torch.gradient(u, spacing=self.dt.item(), dim=1)[0] # t - (t-1)
    # vt = torch.gradient(v, spacing=self.dt.item(), dim=1)[0]
          
    def calculate_terms(self,u,v):
        # BTHW
        with torch.no_grad():
            ux, uy, vx, vy = self.vector_gradients(u,v)

            # approximation for time-step (which we want to learn)
            ut = torch.diff(u, dim=1) / self.dt
            vt = torch.diff(v, dim=1) / self.dt

            # # Material derivatives
            # Du = ut + u*ux + v*uy;
            # Dv = vt + u*vx + v*vy;

            # Coriolis forces
            cor_u = torch.einsum('h,bthw->bthw',self.f,v)
            cor_v = -torch.einsum('h,bthw->bthw',self.f,u)
            
            # # unknown (negative) pressure gradients
            # fix = (Du + cor_u) # divided by density, negative (since this is just the residual directly)
            # fiy = (Dv + cor_v)
            
            # advection terms
            advx = u*ux + v*uy + cor_u
            advy = u*vx + v*vy + cor_v
            
            # same as before (negative pressure gradients) but storing advection terms instead
            fx = (ut + advx[:,:-1])
            fy = (ut + advy[:,:-1])
            
            # so we can recalculate ut, vt using the fx, fy values as opposed to direct estimation (which is lossier / more lossy)
            # ut = fx - advx
            # vt = fy - advy
        
        return fx, fy, advx, advy
    

    def calculate_step(self,u,v):
        fx,fy, advx, advy = self.calculate_terms(u,v) # all past pressures: B (T-1) HW, and all adv terms:  BTHW
        
        fx_shift = self.step(fx,self.Ax)
        fy_shift = self.step(fy,self.Ay) # get shifted pressures, to get the last-step pressure.
                
        # add cohesion loss
        if self.training:
            loss = self.f_cohese * (torch.nn.functional.mse_loss(fx_shift[:,:-1],fx[:,1:]) + torch.nn.functional.mse_loss(fy_shift[:,:-1],fy[:,1:])) # compares (T-2) 
            loss.backward(retain_graph=True) # only if training! #
            
        advx_shift = advx[:,1:]
        advy_shift = advy[:,1:]
         
        ut = fx_shift - advx_shift
        vt = fy_shift - advy_shift # B (T-1) HW, starting at t=1 and ending at t=end (since derivative approx)
        
        u2 = u[:,1:] + ut * self.dt # B (T-1) HW again, but now starting at t=2 and ending at t=end+1
        v2 = v[:,1:] + vt * self.dt
         
        if self.training:
            loss = self.v_cohese * (torch.nn.functional.mse_loss(u2[:,:-1],u[:,2:]) + torch.nn.functional.mse_loss(v2[:,:-1],v[:,2:])) # only compare (T-2) since we start at t=2 now
            loss.backward(retain_graph=True) # only if training! #
            
        return u2[:,-1:], v2[:,-1:] # last timesteps
         
        # with torch.no_grad():   
        # ut = 0
        # vt = 0
        # for _ in range(9):
        #     errut, errvt, u2, v2 = self.t_dot_err(ut,vt,u[:,-1:],v[:,-1:],f2x[:,-1:],f2y[:,-1:])
        #     ut = ut + errut
        #     vt = vt + errvt
        # assert torch.all(torch.isfinite(u2))
        # return u2, v2    
    
    def concentration_resnet(self, c, u, v):
        # BCHW with the exception of the two velocities
        # TODO add source term
        uc = (self.rate[None,:,:,:,0] * u[:,None,:,:]) * c
        vc = (self.rate[None,:,:,:,1] * v[:,None,:,:]) * c
        
        with torch.no_grad():
            cx, cy = self.scalar_gradients(c)
        
        innerx = cx * self.diffusivity[None,:,None,None] - uc
        innery = cy * self.diffusivity[None,:,None,None] - vc
        
        dcdt = self.divergence(innerx, innery)
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
        # BTCHW, c==0 => u, c==1 => v
        u = x[:,:,0,:,:]
        v = x[:,:,1,:,:]
        c = x[:,:,2:,:,:]
        
        c2 = self.concentration_resnet(c,u[:,-1],v[:,-1])
        u2,v2 = self.calculate_step(u,v)

        y = torch.cat([u2[:,:,None,:,:],v2[:,:,None,:,:],c2],dim=2)  
        return y

    
    
    