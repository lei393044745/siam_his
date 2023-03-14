from __future__ import absolute_import

import torch.nn as nn
import torch.nn.functional as F
import torch

__all__ = ['SiamFC']


class SiamFC(nn.Module):

    def __init__(self, out_scale=0.001):
        super(SiamFC, self).__init__()
        self.out_scale = out_scale
    
    def forward(self, z, x):
        out = self._fast_xcorr(torch.cat(z,dim=0), torch.cat(x,dim=0)) * self.out_scale
        
        res = []
        for i in range(1,6):
            res.append(out[(i-1)*3:i*3,:,:,:])
            
        return sum(res)
    
    def _fast_xcorr(self, z, x):
        # fast cross correlation
        nz = z.size(0)
        nx, c, h, w = x.size()
        x = x.view(-1, nz * c, h, w)
        out = F.conv2d(x, z, groups=nz)
        out = out.view(nx, -1, out.size(-2), out.size(-1))
        return out

