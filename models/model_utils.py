import torch
import torch.nn as nn
import math

# 2D convolution with kernel size 1, used after feature stacking, WxHxC => WxHx(C/2)
def conv_1x1(in_channels, out_channels, n_type):
    if (n_type == 'elu'):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False),
            nn.ELU(inplace=False)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False)
        )

class Interpolate(nn.Module):
    def __init__(self, size, mode):
        super(Interpolate, self).__init__()
        # size: expected size after interpolation
        # mode: interpolation type (e.g. bilinear, nearest)

        self.interp = nn.functional.interpolate
        self.size = size
        self.mode = mode
        
    def forward(self, x):
        out = self.interp(x, size=self.size, mode=self.mode) #, align_corners=False
        
        return out

class PartialConv2d(torch.nn.Conv2d):
    def __init__(self, *args, **kwargs):

        # whether the mask is multi-channel or not
        if 'multi_channel' in kwargs:
            self.multi_channel = kwargs['multi_channel']
            kwargs.pop('multi_channel')
        else:
            self.multi_channel = False  

        if 'return_mask' in kwargs:
            self.return_mask = kwargs['return_mask']
            kwargs.pop('return_mask')
        else:
            self.return_mask = False

        super(PartialConv2d, self).__init__(*args, **kwargs)

        if self.multi_channel:
            self.weight_maskUpdater = torch.ones(self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1])
        else:
            self.weight_maskUpdater = torch.ones(1, 1, self.kernel_size[0], self.kernel_size[1])
            
        self.slide_winsize = self.weight_maskUpdater.shape[1] * self.weight_maskUpdater.shape[2] * self.weight_maskUpdater.shape[3]

        self.last_size = (None, None)
        self.update_mask = None
        self.mask_ratio = None

    def forward(self, input, mask_in=None):
        
        if mask_in is not None or self.last_size != (input.data.shape[2], input.data.shape[3]):
            self.last_size = (input.data.shape[2], input.data.shape[3])

            with torch.no_grad():
                if self.weight_maskUpdater.type() != input.type():
                    self.weight_maskUpdater = self.weight_maskUpdater.to(input)

                if mask_in is None:
                    # if mask is not provided, create a mask
                    if self.multi_channel:
                        mask = torch.ones(input.data.shape[0], input.data.shape[1], input.data.shape[2], input.data.shape[3]).to(input)
                    else:
                        mask = torch.ones(1, 1, input.data.shape[2], input.data.shape[3]).to(input)
                else:
                    mask = mask_in
                        
                self.update_mask = torch.nn.functional.conv2d(mask, self.weight_maskUpdater, bias=None, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=1)

                self.mask_ratio = self.slide_winsize/(self.update_mask + 1e-8)
                self.update_mask = torch.clamp(self.update_mask, 0, 1)
                self.mask_ratio = torch.mul(self.mask_ratio, self.update_mask)

        raw_out = super(PartialConv2d, self).forward(torch.mul(input, mask) if mask_in is not None else input)

        if self.bias is not None:
            bias_view = self.bias.view(1, self.out_channels, 1, 1)
            output = torch.mul(raw_out - bias_view, self.mask_ratio) + bias_view
            output = torch.mul(output, self.update_mask)
        else:
            output = torch.mul(raw_out, self.mask_ratio)


        if self.return_mask:
            return output, self.update_mask
        else:
            return output

class PartialConv2dBlock(torch.nn.Module):
    def __init__(self, in_ch, out_ch, bn=False, sample='none-3', activ='elu',
                 conv_bias=False):
        super().__init__()
        if sample == 'down-5':
            self.conv = PartialConv2d(in_ch, out_ch, 5, stride=2, padding=2,\
                bias=conv_bias, return_mask=True)
        elif sample == 'down-7':
            self.conv = PartialConv2d(in_ch, out_ch, 7, stride=2, padding=3,\
                bias=conv_bias, return_mask=True)
        elif sample == 'down-3':
            self.conv = PartialConv2d(in_ch, out_ch, 3, stride=2, padding=1,\
                bias=conv_bias, return_mask=True)
        elif sample== 'none-7':            
            self.conv = PartialConv2d(in_ch, out_ch, 7, stride=1, padding=3,\
                bias=conv_bias, return_mask=True)
        elif sample== 'none-5':            
            self.conv = PartialConv2d(in_ch, out_ch, 5, stride=1, padding=2,\
                bias=conv_bias, return_mask=True)
        elif sample== 'none-1':
            self.conv = PartialConv2d(in_ch, out_ch, 1, stride=1, padding=0,\
                bias=conv_bias, return_mask=True)
        else:
            self.conv = PartialConv2d(in_ch, out_ch, 3, stride=1, padding=1,\
                bias=conv_bias, return_mask=True)

        if bn and activ != "elu":
            self.bn = torch.nn.BatchNorm2d(out_ch)
        if activ == 'relu':
            self.activation = torch.nn.ReLU(inplace=True)
        elif activ == 'leaky':
            self.activation = torch.nn.LeakyReLU(negative_slope=0.2, inplace=True)
        elif activ == "elu":
            self.activation = torch.nn.ELU(inplace=True)

    def forward(self, input, input_mask):
        h, h_mask = self.conv(input, input_mask)
        if hasattr(self, 'bn'):
            h = self.bn(h)
        if hasattr(self, 'activation'):
            h = self.activation(h)
        return h, h_mask

def partial_conv2d_preactivation(in_channels, out_channels, n_type='elu'):
    if (n_type == 'elu'):
        return  torch.nn.ELU(inplace=False),\
            PartialConv2d(in_channels, out_channels, 3, stride=1, padding=1,
                bias=False, return_mask=True)
    else:
        return torch.nn.BatchNorm2d(in_channels),\
            torch.nn.ReLU(inplace=False),\
            PartialConv2d(in_channels, out_channels, 3, stride=1, padding=1,\
                bias=False, return_mask=True)

class PreActivatedPartialResidualConv2d(torch.nn.Module):
    def __init__(self, ndf, activ='batch_norm'):
        super(PreActivatedPartialResidualConv2d, self).__init__()
        # ndf: constant number from channels
        
        self.activ = activ
        if self.activ == 'batch_norm':
            self.activ1, self.bn1, self.conv1 = partial_conv2d_preactivation(\
                ndf, ndf, n_type=activ)
            self.activ2, self.bn2, self.conv2 = partial_conv2d_preactivation(\
                ndf, ndf, n_type=activ)
        else:
            self.activ1, self.conv1 = partial_conv2d_preactivation(ndf, ndf)
            self.activ2, self.conv2 = partial_conv2d_preactivation(ndf, ndf)

    def forward(self, input_data, input_mask):
        residual = input_data
        if self.activ == 'batch_norm':
            out, out_mask = self.conv1(self.activ1(self.bn1(input_data)), input_mask)
            out, out_mask = self.conv2(self.activ2(self.bn2(out)), out_mask)
        else:
            out, out_mask = self.conv1(self.activ1(input_data), input_mask)
            out, out_mask = self.conv2(self.activ2(out), out_mask)
        out += residual

        return out, out_mask