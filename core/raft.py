import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from update import BasicUpdateBlock, SmallUpdateBlock
from extractor import BasicEncoder, SmallEncoder
from corr import CorrBlock, AlternateCorrBlock
from utils.utils import bilinear_sampler, coords_grid, upflow8
from DexiNed.model import DexiNed

try:
    autocast = torch.cuda.amp.autocast
except:
    # dummy autocast for PyTorch < 1.6
    class autocast:
        def __init__(self, enabled):
            pass
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass


class RAFT(nn.Module):
    def __init__(self, args):
        super(RAFT, self).__init__()
        self.args = args
       
        dexi_path = 'core/DexiNed/checkpoints/14_model.pth'
        dexi = DexiNed()
        dexi.load_state_dict(torch.load(dexi_path))
        self.dexined = dexi
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        

        if args.small:
            self.hidden_dim = hdim = 96
            self.context_dim = cdim = 64
            args.corr_levels = 4
            args.corr_radius = 3
        
        else:
            self.hidden_dim = hdim = 128
            self.context_dim = cdim = 128
            args.corr_levels = 4
            args.corr_radius = 4

        if 'dropout' not in self.args:
            self.args.dropout = 0

        if 'alternate_corr' not in self.args:
            self.args.alternate_corr = False

        # feature network, context network, and update block
        if args.small:
            self.fnet = SmallEncoder(input_dim=3, output_dim=128, norm_fn='instance', dropout=args.dropout)        
            self.cnet = SmallEncoder(input_dim=3, output_dim=hdim+cdim, norm_fn='none', dropout=args.dropout)
            self.update_block = SmallUpdateBlock(self.args, hidden_dim=hdim)
            
            self.efnet = SmallEncoder(input_dim=7, output_dim=128, norm_fn='instance', dropout=args.dropout)        
            self.ecnet = SmallEncoder(input_dim=7, output_dim=hdim+cdim, norm_fn='none', dropout=args.dropout)
            

        else:
            self.fnet = BasicEncoder(input_dim=3, output_dim=256, norm_fn='instance', dropout=args.dropout)        
            self.cnet = BasicEncoder(input_dim=3, output_dim=hdim+cdim, norm_fn='batch', dropout=args.dropout)
            self.update_block = BasicUpdateBlock(self.args, hidden_dim=hdim)
            
            self.efnet = BasicEncoder(input_dim=7, output_dim=256, norm_fn='instance', dropout=args.dropout)        
            self.ecnet = BasicEncoder(input_dim=7, output_dim=hdim+cdim, norm_fn='batch', dropout=args.dropout)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H//8, W//8).to(img.device)
        coords1 = coords_grid(N, H//8, W//8).to(img.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    def upsample_flow(self, flow, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3,3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 8*H, 8*W)


    def forward(self, image1, image2, iters=12, flow_init=None, upsample=True, test_mode=False):
        """ Estimate optical flow between pair of frames """

        image1 = 2 * (image1 / 255.0) - 1.0
        image2 = 2 * (image2 / 255.0) - 1.0

        image1 = image1.contiguous()
        image2 = image2.contiguous()
        #print(' the size of image1: ', image1.shape)
        #print('the size of image 1 is: ', image1.shape)
        with torch.no_grad():
            edge1 = self.dexined(image1) # Bx7xHxW
            edge2 = self.dexined(image2)        
        
        edge1 = torch.stack(edge1)        
        edge2 = torch.stack(edge2)
        #print(' the type of edge1: ', type(edge1), edge1.shape)
        #pause
        em1 = torch.squeeze(edge1,2)
        em2 = torch.squeeze(edge2,2)        
        #if len(em1.shape) ==4:
        em1 = em1.permute(1, 0, 2, 3)
        em2 = em2.permute(1, 0, 2, 3)
        #print(' the type of image1: ', type(image1), image1.shape)
        #print(' the type of em1: ', type(em1), em1.shape)
        #imedge1 = torch.cat([image1, em1], dim=1)
        #imedge2 = torch.cat([image2, em2], dim=1)
        #print(' the type of em1: ', type(imedge1), imedge1.shape)
        #pause
        hdim = self.hidden_dim
        cdim = self.context_dim

        # run the feature network
        with autocast(enabled=self.args.mixed_precision):
            fmap1, fmap2 = self.fnet([image1, image2])
            fem1, fem2 = self.efnet([em1, em2])                
        
      
        fmap1 = fmap1.float()
        fmap2 = fmap2.float()
        fem1 = fem1.float()
        fem2 = fem2.float()
        if self.args.alternate_corr:
            corr_fn = AlternateCorrBlock(fmap1, fmap2, radius=self.args.corr_radius)
            corr_en = AlternateCorrBlock(fem1, fem2, radius=self.args.corr_radius)
        else:
            corr_fn = CorrBlock(fmap1, fmap2, radius=self.args.corr_radius)
            corr_en = CorrBlock(fem1, fem2, radius=self.args.corr_radius)

        # run the context network
        with autocast(enabled=self.args.mixed_precision):
            cnet = self.cnet(image1)
            enet = self.ecnet(em1)
            net, inp = torch.split(cnet, [hdim, cdim], dim=1)
            enet, einp = torch.split(enet, [hdim, cdim], dim=1)
            net = torch.tanh(net)
            inp = torch.relu(inp)
            enet = torch.tanh(enet)
            einp = torch.relu(einp)
            

        coords0, coords1 = self.initialize_flow(image1)
        ecoords0, ecoords1 = self.initialize_flow(em1)

        if flow_init is not None:
            coords1 = coords1 + flow_init

        flow_predictions = []
        for itr in range(iters):
            coords1 = coords1.detach()
            ecoords1 = ecoords1.detach()
            corr = corr_fn(coords1) # index correlation volume
            ecorr = corr_en(ecoords1)

            flow = coords1 - coords0
            eflow = ecoords1 - ecoords0
            #flow = flow + eflow
            with autocast(enabled=self.args.mixed_precision):
                net, up_mask, delta_flow = self.update_block(net, inp, corr, flow)
                enet, eup_mask, delta_eflow= self.update_block(enet,einp, ecorr, eflow)

            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow + delta_eflow
            ecoords1 = ecoords1 + delta_eflow

            # upsample predictions
            if up_mask is None:
                flow_up = upflow8(coords1 - coords0)
            else:
                flow_up = self.upsample_flow(coords1 - coords0, up_mask)
            
            flow_predictions.append(flow_up)

        if test_mode:
            return coords1 - coords0, flow_up
            
        return flow_predictions
