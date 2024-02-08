__author__ = 'asadam'

import torch
import torch.nn as nn
import torch.nn.functional as F


# problem with all these approaches is that the mode matrix (U, V^t for USV^T) or (Q for QAQ^-1) can be arbitrarily scrambled
# this scrambling leads to useless forward stepping... we need to enforce some kind of ordering on the modes
# Currently the ordering is enforced by eigenvalue magnitude,
# but this keeps changing (unstable), and is not guaranteed to be consistent across mode-sets.

# using QR helps with unscrambling, but magnitude is stil -> inf...

class Operator(nn.Module):
    def __init__(self, nlatent, ntime, h, w, device, n_embd=100, nlayers=2, **kwargs):
        super(Operator, self).__init__()
        
        self.nlatent = nlatent
        self.ntime = ntime
        self.nlayers = nlayers
        self.n_embd = n_embd
        self.h = h
        self.w = w
       
        self.m = ntime // 2
        
        self.w = nn.Parameter(torch.tensor([1.0],device=device))
        self.device = device
        
    def POD(self,x):
        with torch.no_grad():            
            vssv = torch.einsum("Bt..., BT... -> BtT", x, x) # X^T X, shape BtT, results in vs^t s v^t
            # e, v = torch.linalg.eigh(vssv) # shapes are Bt, BtT #+ self.reg[None,:,:]
            _, e, vt = torch.linalg.svd(vssv)
            # print(e[0])
            e = torch.where(e > 1, e, 1)
            s = torch.sqrt(e)
            # einv = torch.diag_embed(torch.reciprocal(e),offset=0,dim1=-2,dim2=-1)
            sinv = torch.diag_embed(torch.reciprocal(s),offset=0,dim1=-2,dim2=-1)
            ####
            # want s^-1 v^t v s^t s v^t = s v^t
            # svt = torch.einsum("Bm, Bmn -> Bmn", s, vt)
            ####
            # s = torch.sqrt(e) # note should be real since vssv is positive semi-definite and symmetric
            assert torch.all(torch.isfinite(sinv)) ,"38"
            u = torch.einsum("Bt..., BTt, BTm -> Bm...", x, vt, sinv)
            assert torch.all(torch.isfinite(u)), "40"
        return u, vt, sinv
    
    # def POD_phi(self,x):
    #     with torch.no_grad():            
    #         vssv = torch.einsum("Btmchw, BTmchw -> BtT", x, x) # X^T X, shape BtT, results in vs^t s v^t
    #         _, e, vt = torch.linalg.svd(vssv)
    #         e = torch.where(e > 1, e, 1)
    #         s = torch.sqrt(e)
    #         sinv = torch.diag_embed(torch.reciprocal(s),offset=0,dim1=-2,dim2=-1)
    #         u = torch.einsum("Btachw, BTt, BTm -> Bmachw", x, vt, sinv)
        
    #     return u, vt, sinv
    
    def POD_phi(self,x):
        with torch.no_grad():            
            vssv = torch.einsum("Bt..., BT... -> BtT", x, x) # X^T X, shape BtT, results in vs^t s v^t
            _, e, vt = torch.linalg.svd(vssv)
            e = torch.where(e > 1, e, 1)
            s = torch.sqrt(e)
            sinv = torch.complex(torch.diag_embed(torch.reciprocal(s),offset=0,dim1=-2,dim2=-1),torch.zeros(1,device=self.device))
            
            assert torch.all(torch.isfinite(sinv))
            
            ua = torch.einsum("Bt..., BTt, BTm -> Bm...", x, vt, sinv)
            
            uk = ua.reshape(ua.shape[0],ua.shape[1],-1)#.permute(0,2,1)
            # orthonormalize columns
            # uq, _ = torch.linalg.qr(uk)
            _,uq = torch.linalg.qr(uk)
            u = uq.reshape(ua.shape)#.permute(0,2,1)
            
            assert torch.all(torch.isfinite(u))
            
            svinv = torch.linalg.pinv(torch.einsum("Bt..., BT... -> BtT", u, x)) # u^t x = s^t v^t x = sv^t, and then we invert.
            
        return u, svinv
    
    
    def DMD_phi(self, x, y):
        # x has shape BTchw
        u, svinv = self.POD_phi(x)
        A_tilde = torch.einsum("Bm..., Bt..., Btc -> Bmc", u, y, svinv) # notice flip for v^t (since we want v)
        # print(A_tilde.shape)
        assert torch.all(torch.isfinite(A_tilde)),"76"
        # diagonalize to get modes
        # L, V = torch.linalg.eig(A_tilde)
        # # limit eigenvalue magnitude
        # # L = torch.where(torch.abs(L) > 5, L, L/torch.abs(L))
        # # compute A_tilde again
        # A_tilde = torch.einsum("Bmn, Bn, Bnp -> Bmp", V, L, torch.linalg.inv(V))
        
        return u, A_tilde
    
    def apply_phi(self, u, A_tilde, x):
        return torch.einsum("Bmachw, BMm, BMACHW, BACHW -> Bachw", u, A_tilde, u, x) # change QK to match new dimensions
    
    def DMD_s(self, x, y):
        # x has shape BTm
        # so a = yx^pinv
        A = torch.linalg.lstsq(x, y).solution # btm1, btm2 -> bm1m2
        
        return A
    
    def apply_s(self, A, x):
        return torch.einsum("BmM,Bm -> BM", A, x) # note a is transposed due to lstsq
        
    
    # def DMD(self, x, y):
    #     # x has shape BTchw
    #     u, vt, sinv = self.POD(x)
    #     A_tilde = torch.einsum("Bmchw, Btchw, BTt, BTa -> Bma", u, y, vt, sinv) # notice flip for v^t (since we want v)
        
    #     at_u, at_s, at_vt = torch.linalg.svd(A_tilde)
        
    #     phi_left = torch.einsum("Bmchw, Bmt -> Btchw", u, at_u)
    #     phi_right = torch.einsum("Btchw, Bmt -> Bmchw", u, at_vt)
        
    #     # # diagonalize to get modes
    #     # L, V = torch.linalg.eig(A_tilde)
    #     # # compute modes
    #     # phi = torch.einsum("Bmchw, Bmt -> Btchw", u, V)

    #     return phi_left, at_s, phi_right
    
    def DMD(self, x, y):
        # x has shape BTchw
        
        # x[~torch.isfinite(x)] = 0
        # y[~torch.isfinite(y)] = 0
        
        
        u, vt, sinv = self.POD(x)
        A_tilde = torch.einsum("Bmchw, Btchw, BTt, BTa -> Bma", u, y, vt, sinv) # notice flip for v^t (since we want v)
        
        # at_u, at_s, at_vt = torch.linalg.svd(A_tilde)
        assert torch.all(torch.isfinite(y)),"124"
        assert torch.all(torch.isfinite(vt)),"125"
        # phi_left = torch.einsum("Bmchw, Bmt -> Btchw", u, at_u)
        # phi_right = torch.einsum("Btchw, Bmt -> Bmchw", u, at_vt)
        assert torch.all(torch.isfinite(A_tilde)),"126"
        # diagonalize to get modes
        L, V = torch.linalg.eig(A_tilde)
        # compute modes
        phi_left = torch.einsum("Bmchw, Bmt -> Btchw", torch.complex(u,torch.tensor([0.0],device=self.device)), V)
        phi_right = torch.einsum("Bmchw, Btm -> Btchw", torch.complex(u,torch.tensor([0.0],device=self.device)), torch.linalg.inv(V))
        # phi = V

        return phi_left, L, phi_right
    
    # def apply(self, phi_left, s, phi_right, x):
    #     return torch.einsum("Bmchw, Bm, BmCHW, BtCHW -> Btchw", phi_left, s, phi_right, x)
    #     # return torch.einsum("Bmchw, Bm, BmCHW, BCHW -> Bchw", phi, L, phi, x).real
    def apply(self, phi_left, s, phi_right, x):
        x_complex = torch.complex(x,torch.tensor([0.0],device=self.device))
        return torch.einsum("Bmchw, Bm, BmCHW, BtCHW -> Btchw", phi_left, s, phi_right, x_complex).real
    
    # def compose(self, x_complete):
    #     with torch.no_grad():
    #         phi_lefts = []
    #         s = []
    #         phi_rights = []
    #         for i in range(self.m):
    #             x = x_complete[:,i:i+self.m,:,:,:]
    #             y = x_complete[:,i+1:i+self.m+1,:,:,:]
                
    #             phi_left, s_, phi_right = self.DMD(x, y)
    #             # print(i)
    #             phi_lefts.append(phi_left)
    #             s.append(s_)
    #             phi_rights.append(phi_right)
                
    #         phi_lefts = torch.stack(phi_lefts, dim=1) # Btmchw (t is which phi_left)
    #         mu_phi_left = torch.mean(phi_lefts, dim=1)
    #         phi_lefts = phi_lefts - mu_phi_left[:,None,:,:,:,:]
    #         uphi_left, Aphi_left = self.DMD_phi(phi_lefts[:,:-1], phi_lefts[:,1:])
    #         phi_left_last = phi_lefts[:,-1]
            
    #         del phi_lefts
                        
    #         phi_rights = torch.stack(phi_rights, dim=1)
    #         mu_phi_right = torch.mean(phi_rights, dim=1)
    #         phi_rights = phi_rights - mu_phi_right[:,None,:,:,:,:]
    #         uphi_right, Aphi_right = self.DMD_phi(phi_rights[:,:-1], phi_rights[:,1:])
    #         phi_right_last = phi_rights[:,-1]
            
    #         del phi_rights
            
    #         s = torch.stack(s, dim=1) # Btm
    #         mu_s = torch.mean(s, dim=1)
    #         s = s - mu_s[:,None,:]
    #         a_s = self.DMD_s(s[:,:-1], s[:,1:])
    #         s_last = s[:,-1]
            
    #         return uphi_left, Aphi_left, phi_left_last, mu_phi_left, uphi_right, Aphi_right, phi_right_last, mu_phi_right, a_s, s_last, mu_s
        
            
    # def produce(self, uphi_left, Aphi_left, phi_left_last, mu_phi_left, uphi_right, Aphi_right, phi_right_last, mu_phi_right, a_s, s_last, mu_s):
        
    #     with torch.no_grad():
            
    #         phi_left = mu_phi_left #+ self.apply_phi(uphi_left, Aphi_left, phi_left_last) 
    #         phi_right = mu_phi_right + self.apply_phi(uphi_right, Aphi_right, phi_right_last)
    #         s = mu_s #+ self.apply_s(a_s, s_last)
    #         # # print(s.shape)
    #     return phi_left, s, phi_right
    
    def compose(self, x_complete):
        with torch.no_grad():
            phi_lefts = []
            s = []
            phi_rights = []
            for i in range(self.m):
                x = x_complete[:,i:i+self.m,:,:,:]
                y = x_complete[:,i+1:i+self.m+1,:,:,:]
                # print(i+1,i+self.m+1, x_complete.shape[1])
                
                phi_left, s_, phi_right = self.DMD(x, y)
                # print(i+1,i+self.m+1, x_complete.shape[1],"after")
                # print(i)
                phi_lefts.append(phi_left)
                phi_rights.append(phi_right)
                s.append(s_)
            
            # print("stacking")
            phi_lefts = torch.stack(phi_lefts, dim=1) # Btmchw (t is which phi_left)
            # mu_phi_left = torch.mean(phi_lefts, dim=1)
            # phi_lefts = phi_lefts - mu_phi_left[:,None,:,:,:,:]
            uphi_left, Aphi_left = self.DMD_phi(phi_lefts[:,:-1], phi_lefts[:,1:])
            phi_left_last = phi_lefts[:,-1]
            
            del phi_lefts
            
            phi_rights = torch.stack(phi_rights, dim=1) # Btmchw (t is which phi_left)
            # mu_phi_left = torch.mean(phi_lefts, dim=1)
            # phi_lefts = phi_lefts - mu_phi_left[:,None,:,:,:,:]
            uphi_right, Aphi_right = self.DMD_phi(phi_rights[:,:-1], phi_rights[:,1:])
            phi_right_last = phi_rights[:,-1]
            
            del phi_rights
            
            s = torch.stack(s, dim=1) # Btm
            # mu_s = torch.mean(s, dim=1)
            # s = s - mu_s[:,None,:]
            a_s = self.DMD_s(s[:,:-1], s[:,1:])
            s_last = s[:,-1]
            
            
            return uphi_left, Aphi_left, phi_left_last, a_s, s_last, uphi_right, Aphi_right, phi_right_last
            
    def produce(self, uphi_left, Aphi_left, phi_left_last, a_s, s_last, uphi_right, Aphi_right, phi_right_last):
        
        with torch.no_grad():
            
            phi_left = self.apply_phi(uphi_left, Aphi_left, phi_left_last) 
            phi_right = self.apply_phi(uphi_right, Aphi_right, phi_right_last)
            s = s_last#self.apply_s(a_s, s_last)
            # # print(s.shape)
        return phi_left, s, phi_right
    
    # def forward(self,x0): # just a single step
        
    #     parts = self.compose(x0)
    #     phi_left, s, phi_right = self.produce(*parts)
        
    #     return self.apply(phi_left, s, phi_right, x0[:,-1:,:,:,:]) * self.w
        
         
    def forward(self,x0): # just a single step
        
        parts = self.compose(x0)
        phi_left, s, phi_right = self.produce(*parts)

        return self.apply(phi_left, s, phi_right, x0[:,-self.m:,:,:,:]) * self.w  
    
    
    
    
    
    
    
    # def layer(self,x,i):
    #     u = self.POD(x)
        
    #     svt = self.down(u,x)
    #     a = self.Mix(svt,i)
    #     u2 = self.up(a,u)
        
    #     return self.Left(u2, a ,i)   
    
    # def outlayer(self,x):
    #     u = self.POD(x)
    #     svt = self.down(u,x)
    #     a = self.Select(svt)
    #     return self.upf(a,u)[:,None,:,:,:]
          
    # def forward(self,x0): # just a single step
    #     # x has shape BTcHW, and TC = [T0-20]c are to be updated to [T21]c.
        
    #     mu = x0[:,-1:,:,:,:]
    #     # std = (3 * torch.std(x, dim=1, keepdim=True))
    #     # print(std.shape)
    #     # to local
    #     x = x0 - mu
    #     # x2 = x2 / std
    #     # this is okay iff no fixed dimensions remain; i.e. we nondimensionalize every part of the equation!
    #     # but scales are linear and differ per situation
    #     # so this is really a local projection only; we have lost global information with the batch norm...
    #     # which is why the equation must be nondimensionalized completely..
        
    #     for i in range(self.nlayers):
    #         x = self.layer(x,i)

    #     x = self.outlayer(x)
        
    #     return x + mu
        
        # return x1 # B_chw, or BTchw since we want to keep time dimension (but it's just length 1)