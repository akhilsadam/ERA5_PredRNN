__author__ = 'asadam'

import torch
import torch.nn as nn
import torch.nn.functional as F
# import torch_harmonics as th
import math




# 2D incompressible N-S



class Operator(nn.Module):
    def __init__(self, nlatent, ntime, h, w, device, n_embd=400, nlayers=1, activation=torch.nn.ReLU(), **kwargs):
        super(Operator, self).__init__()
        
        self.nlatent = nlatent
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
        
        # dt_scale = 6*3600.0
        # 6370*1000 # Earth's radius (km)
        
        self.lat_rad = pi*lat/180
        self.R_earth = nn.Parameter(torch.tensor([1.0],device=device))

        # Step size as a function of latitude (in m)
        #   Based on 1/25 degree resolution
        self.dx = lambda r: (1/h)*2*pi*r*torch.cos(self.lat_rad)# Radius as a function of latitude
        self.y = lambda dx: torch.cumsum(dx, dim=0)
        self.dt = nn.Parameter(torch.tensor([1e-5],device=device)) # 6-Hourly resolution # TODO make parameter

        # Coriolis parameter
        # self.omega = nn.Parameter(torch.tensor([1.0],device=device)) #7.3e-5 # Rotation rate
         # s^-1 # size h

        # Density
        # self.rho = nn.Parameter([1.204],device=device) # kg/m^3 # assuming incompressible for now! NEED TO CHANGE
        
        # self.rate = nn.Parameter(torch.ones((self.clatent,h,w,2),device=device)) # friction-modulated velocity adjustment
        
        self.diffusivity = nn.Parameter(torch.zeros((self.nlatent),device=device)) # TODO make space-dependent!
        
        n_modes = (ntime//2) - 1
        self.A_v = nn.Parameter(torch.empty((n_modes,n_modes),device=device))
        nn.init.xavier_uniform_(self.A_v.data,gain=0.001)
        # self.Ax = nn.Parameter(torch.empty((n_modes,n_modes),device=device))
        # nn.init.xavier_uniform_(self.Ax.data,gain=0.001)
        # self.Ay = nn.Parameter(torch.empty((n_modes,n_modes),device=device))
        # nn.init.xavier_uniform_(self.Ay.data,gain=0.001)
        
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
        y = torch.einsum("BTHW, BMHW, Mm, Bmhw -> BThw",x,u,A,u) + x # single shift
        return y
    
    def step_w_cat(self,x,A):
        # B(T-1)CHW
        u = self.POD(x) # BMHW
        y = torch.einsum("BtHW, BMHW, Mm, Bmhw -> Bthw",x[:,-1:],u,A,u) + x[:,-1:] # single shift
        full = torch.cat([x,y],dim=1) # BTCHW
        return y
    
    @torch.compile(fullgraph=False)
    def scalar_gradients(self, u, dx, y):
        # Zonal derivative
        # ux = torch.zeros(u.shape,device=self.device)
        # for i in range(self.nlat):
        #     ux[..., i, :] = torch.gradient(u[..., i, :], spacing=dx[i].item(), dim=-1)[0]
        ux = torch.gradient(u, dim=-1)[0] / dx[None, None, :, None]
        # Meridional derivative
        # uy = torch.zeros(u.shape,device=self.device)
        # for i in range(self.nlon):
        #     uy[..., :, i] = torch.gradient(u[..., :, i], spacing=(y,), dim=-1)[0]
        uy = torch.gradient(u, spacing=(y,), dim=-2)[0]
        return ux, uy
    
    @torch.compile(fullgraph=False)
    def vector_gradients(self, u, v, dx, y):
        # Zonal derivative
        # ux = torch.zeros(u.shape,device=self.device)
        # vx = torch.zeros(u.shape,device=self.device)
        # for i in range(self.nlat):
        #     ux[..., i, :] = torch.gradient(u[..., i, :], spacing=dx[i].item(), dim=-1)[0]
        #     vx[..., i, :] = torch.gradient(v[..., i, :], spacing=dx[i].item(), dim=-1)[0]
        ux = torch.gradient(u, dim=-1)[0] / dx[None, None, :, None]
        vx = torch.gradient(v, dim=-1)[0] / dx[None, None, :, None]
            
        # Meridional derivative
        # uy = torch.zeros(u.shape,device=self.device)
        # vy = torch.zeros(u.shape,device=self.device)
        # for i in range(self.nlon):
        #     uy[..., :, i] = torch.gradient(u[..., :, i], spacing=(y,), dim=-1)[0]
        #     vy[..., :, i] = torch.gradient(v[..., :, i], spacing=(y,), dim=-1)[0]
        uy = torch.gradient(u, spacing=(y,), dim=-2)[0]
        vy = torch.gradient(v, spacing=(y,), dim=-2)[0]
        
        return ux, uy, vx, vy

    @torch.compile(fullgraph=False)
    def divergence(self, u, v, dx, y):
        # Zonal derivative
        # ux = torch.zeros(u.shape,device=self.device,requires_grad=True)
        # for i in range(self.nlat):
        #     ux[..., i, :] = torch.gradient(u[..., i, :], spacing=dx[i].item(), dim=-1)[0]
        # ux = torch.stack([torch.gradient(u[..., i, :], spacing=dx[i].item(), dim=-1)[0] for i in range(self.nlat)], dim=-2)
        ux = torch.gradient(u, dim=-1)[0] / dx[None, None, :, None] # can divide afterward since change is perpendicular to gradient direction
            
        # Meridional derivative
        # vy = torch.zeros(u.shape,device=self.device,requires_grad=True)
        # for i in range(self.nlon):
        #     vy[..., :, i] = torch.gradient(v[..., :, i], spacing=(y,), dim=-1)[0]
        # vy = torch.stack([torch.gradient(v[..., :, i], spacing=(y,), dim=-1)[0] for i in range(self.nlon)], dim=-1)
        vy = torch.gradient(v, spacing=(y,), dim=-2)[0]
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
          
    # def calculate_terms(self, u, v, dx, y):
    #     # BTHW
    #     with torch.no_grad():
    #         ux, uy, vx, vy = self.vector_gradients(u, v, dx, y)

    #         # approximation for time-step (which we want to learn)
    #         ut = torch.diff(u, dim=1) / self.dt
    #         vt = torch.diff(v, dim=1) / self.dt

    #         # # Material derivatives
    #         # Du = ut + u*ux + v*uy;
    #         # Dv = vt + u*vx + v*vy;

    #         # Coriolis forces
    #         f = 2*self.omega*torch.sin(self.lat_rad)
    #         cor_u = torch.einsum('h,bthw->bthw',f,v)
    #         cor_v = -torch.einsum('h,bthw->bthw',f,u)
            
    #         # # unknown (negative) pressure gradients
    #         # fix = (Du + cor_u) # divided by density, negative (since this is just the residual directly)
    #         # fiy = (Dv + cor_v)
            
    #         # advection terms
    #         advx = u*ux + v*uy + cor_u
    #         advy = u*vx + v*vy + cor_v
            
    #         # same as before (negative pressure gradients) but storing advection terms instead
    #         fx = (ut + advx[:,:-1])
    #         fy = (vt + advy[:,:-1])
            
    #         # so we can recalculate ut, vt using the fx, fy values as opposed to direct estimation (which is lossier / more lossy)
    #         # ut = fx - advx
    #         # vt = fy - advy
        
    #     return fx, fy, advx, advy
    

    # def calculate_step(self, u, v, dx, y):
    #     fx,fy, advx, advy = self.calculate_terms(u, v, dx, y) # all past pressures: B (T-1) HW, and all adv terms:  BTHW
        
    #     fx_shift = self.step(fx,self.Ax)
    #     fy_shift = self.step(fy,self.Ay) # get shifted pressures, to get the last-step pressure.
                
    #     # # add cohesion loss
    #     # if self.training:
    #     #     loss = self.f_cohese * (torch.nn.functional.mse_loss(fx_shift[:,:-1],fx[:,1:]) + torch.nn.functional.mse_loss(fy_shift[:,:-1],fy[:,1:])) # compares (T-2) 
    #     #     loss.backward(retain_graph=True) # only if training! #
            
    #     advx_shift = advx[:,1:]
    #     advy_shift = advy[:,1:]
         
    #     ut = fx_shift - advx_shift
    #     vt = fy_shift - advy_shift # B (T-1) HW, starting at t=1 and ending at t=end (since derivative approx)
        
    #     u2 = u[:,1:] + ut * self.dt # B (T-1) HW again, but now starting at t=2 and ending at t=end+1
    #     v2 = v[:,1:] + vt * self.dt
         
    #     # if self.training:
    #     #     loss = self.v_cohese * (torch.nn.functional.mse_loss(u2[:,:-1],u[:,2:]) + torch.nn.functional.mse_loss(v2[:,:-1],v[:,2:])) # only compare (T-2) since we start at t=2 now
    #     #     loss.backward(retain_graph=True) # only if training! #
            
    #     return u2[:,-1:], v2[:,-1:] # last timesteps
         
    #     # with torch.no_grad():   
    #     # ut = 0
    #     # vt = 0
    #     # for _ in range(9):
    #     #     errut, errvt, u2, v2 = self.t_dot_err(ut,vt,u[:,-1:],v[:,-1:],f2x[:,-1:],f2y[:,-1:])
    #     #     ut = ut + errut
    #     #     vt = vt + errvt
    #     # assert torch.all(torch.isfinite(u2))
    #     # return u2, v2    
        
    def batch_lstsq(self, cx, cy, a):
        # B(T-2)CHW 
        t_half = cx.shape[1] // 2
        cx = cx.reshape((cx.shape[0],t_half,2,*cx.shape[2:])).permute(0,1,3,4,5,2) # break into sets of 2
        cy = cy.reshape((cy.shape[0],t_half,2,*cy.shape[2:])).permute(0,1,3,4,5,2)
        
        C_mat = torch.stack([cx,cy],axis=-1) # stack on n, for (..., m={t1,t2}, n={x,y}) -> BtCHW,2,2
        # expect a to be reshaped as well 
        a = a.reshape(a.shape[0],t_half,2,*a.shape[2:]).permute(0,1,3,4,5,2)[...,None] # BtCHW,2,1
        # underdetermined solution?
        v = torch.linalg.lstsq(C_mat,a)[0][...,0] # BtCHW,2,1 -> BtCHW,2
        return v
    
    def calculate_adv(self, v, cx, cy):
        # expand v first to BTCHW2, full size now
        v = torch.repeat_interleave(v,repeats=2,dim=1)
        return v[...,0] * cx + v[...,1] * cy

        
        
        
    def neuralvelocity(self, c, dx, y):
        # BTCHW
        cx, cy = self.scalar_gradients(c, dx, y)
        
        innerx = cx * self.diffusivity[None,:,None,None]
        innery = cy * self.diffusivity[None,:,None,None]
        
        div_term = self.divergence(innerx, innery, dx, y)
        
        ct = torch.diff(c, dim=1) / self.dt
        
        a = ct - div_term[:,:-1] # B (T-1) CHW # v_dot_grad_c
        
        # find velocity that satisfies Occam's razor using cx, cy for gradients
        # basically have it change every two steps
        v = self.batch_lstsq(cx[:,:-2], cy[:,:-2], a[:,:-1]) # B (T-2)
        
        return div_term, v, cx, cy
        
    
    def concentration_resnet(self, c, div_term, v_dot_grad_c):
        # BTCHW (expects u,v in form BTCHW)
        # advection-diffusion-reaction
        # TODO add reaction        


        dcdt = div_term - v_dot_grad_c # assumes incompressible v ! # TODO add this assumption to the stepper for v
        # assert torch.all(torch.isfinite(dcdt))
        c2 = c + dcdt * self.dt   
        # assert torch.all(torch.isfinite(c2))
        # # add cohesion loss
        if self.training:
            loss = self.c_cohese * torch.nn.functional.mse_loss(c2[:,:-1],c[:,1:])
            loss.backward(retain_graph=True) # only if training!
        
        return c2[:,-1:]
    
    def forward(self, c):
        assert c.shape[1] > 2, "Not enough history. Coherency losses will fail"
        
        # make grid
        dx = self.dx(self.R_earth)
        y = self.y(dx)
        
        div_term, v_history, cx, cy = self.neuralvelocity(c, dx, y)
        v = self.step_w_cat(v_history, A_v)
        v_dot_grad_c = self.calculate_adv(v, cx, cy)
        c2 = self.concentration_resnet(c, div_term, v_dot_grad_c)

        return y

    
    
    