import torch

from .model_utils import *

class PartialNet(nn.Module):
    def __init__(self,  width, height, ndf, dilation=1, normal_type='batch_norm', upsample='nearest'):
        super(PartialNet, self).__init__()
        # width: input width
        # height: input height
        # ndf: constant number from channels
        # dil: dilation value - parameter for convolutional layers
        # norma_type: normalization type (elu | batch norm)
        # upsample: upsampling type (nearest | bilateral)

        self.h = height
        self.w = width
        self.dil = dilation
        self.type = normal_type
        self.upsample = upsample

        # encoder
        self.encoder_conv1_1 = PartialConv2dBlock(1, ndf, bn=False, activ='elu', sample='none-7')
        self.encoder_conv1_2 = PartialConv2dBlock(ndf, ndf * 2, bn=False, activ='elu', sample='none-5')
        self.encoder_conv2_1 = PartialConv2dBlock(ndf * 2, ndf * 4, bn=False, activ='elu', sample='down-3')
        self.encoder_conv2_2 = PartialConv2dBlock(ndf * 4, ndf * 4, bn=False, activ='elu')
        self.encoder_conv2_3 = PartialConv2dBlock(ndf * 4, ndf * 4, bn=False, activ='elu')
        self.encoder_conv3_1 = PartialConv2dBlock(ndf * 4, ndf * 8, bn=False, activ='elu', sample='down-3')
        self.encoder_conv3_2 = PartialConv2dBlock(ndf * 8, ndf * 8, bn=False, activ='elu')
        self.encoder_conv3_3 = PartialConv2dBlock(ndf * 8, ndf * 8, bn=False, activ='elu')

        # bottleneck
        self.encoder_conv4 = PartialConv2dBlock(ndf * 8, ndf * 8, bn=False, activ='elu', sample='down-3')
        self.encoder_resblock1 = PreActivatedPartialResidualConv2d(ndf * 8, activ='elu')
        self.encoder_resblock2 = PreActivatedPartialResidualConv2d(ndf * 8, activ='elu')

        # masks
        self.mask_downsampled1 = Interpolate((self.h, self.w), mode='nearest')
        self.mask_downsampled2 = Interpolate((self.h // 2, self.w // 2), mode='nearest')
        self.mask_downsampled3 = Interpolate((self.h // 4, self.w // 4), mode='nearest')
        self.mask_downsampled4 = Interpolate((self.h // 8, self.w // 8), mode='nearest')


        # decoder
        self.decoder_upsample3 = Interpolate((self.h // 4, self.w // 4), mode=self.upsample)
        self.decoder_deconv4 = PartialConv2dBlock(ndf * 8, ndf * 8, bn=False, activ='elu')
        self.decoder_conv_id_3 = conv_1x1(2 * ndf * 8, ndf * 8, n_type=self.type)
        self.decoder_deconv3_3 = PartialConv2dBlock(ndf * 8, ndf * 8, bn=False, activ='elu')
        self.decoder_deconv3_2 = PartialConv2dBlock(ndf * 8, ndf * 8, bn=False, activ='elu')
        self.decoder_upsample2 = Interpolate((self.h // 2, self.w // 2), mode=self.upsample)
        self.decoder_deconv3_1 = PartialConv2dBlock(ndf * 8, ndf * 4, bn=False, activ='elu')
        self.decoder_conv_id_2 = conv_1x1(2 * ndf * 4, ndf * 4, n_type=self.type)
        self.decoder_deconv2_3 = PartialConv2dBlock(ndf * 4, ndf * 4, bn=False, activ='elu')
        self.decoder_deconv2_2 = PartialConv2dBlock(ndf * 4, ndf * 4, bn=False, activ='elu')
        self.decoder_upsample1 = Interpolate((self.h, self.w), mode=self.upsample)
        self.decoder_deconv2_1 = PartialConv2dBlock(ndf * 4, ndf * 2, bn=False, activ='elu')
        self.decoder_conv_id_1 = conv_1x1(2 * ndf * 2, ndf * 2, n_type=self.type)
        self.decoder_deconv1_2 = PartialConv2dBlock(ndf * 2, ndf, bn=False, activ='elu')
        self.decoder_deconv1_1 = PartialConv2dBlock(ndf, 1, bn=False, activ='no_acitv')

    def forward(self, x, mask):
        out, m = self.encoder_conv1_1(x, mask)
        out_pre_ds_1, m = self.encoder_conv1_2(out, m)
        out, m = self.encoder_conv2_1(out_pre_ds_1, m)
        out, m = self.encoder_conv2_2(out, m)
        out_pre_ds_2, m = self.encoder_conv2_3(out, m)
        out, m = self.encoder_conv3_1(out_pre_ds_2, m)
        out, m = self.encoder_conv3_2(out, m)
        out_pre_ds_3, m = self.encoder_conv3_3(out, m)
        out, m = self.encoder_conv4(out_pre_ds_3, m)
        out, m = self.encoder_resblock1(out, m)
        out, m = self.encoder_resblock2(out,m)
        out = self.decoder_upsample3(out)
        mask = self.mask_downsampled3(m)
        out, m = self.decoder_deconv4(out, mask)
        out_post_up_3 = torch.cat((out, out_pre_ds_3), 1)
        out = self.decoder_conv_id_3(out_post_up_3)
        out, m = self.decoder_deconv3_3(out, m)
        out, m = self.decoder_deconv3_2(out, m)
        out = self.decoder_upsample2(out)
        mask = self.mask_downsampled2(m)
        out, m = self.decoder_deconv3_1(out, mask)
        out_post_up_2 = torch.cat((out, out_pre_ds_2), 1)
        out = self.decoder_conv_id_2(out_post_up_2)
        out, m = self.decoder_deconv2_3(out, m)
        out, m = self.decoder_deconv2_2(out, m)
        out = self.decoder_upsample1(out)
        mask = self.mask_downsampled1(m)
        out, m = self.decoder_deconv2_1(out, mask)
        out_post_up_1 = torch.cat((out, out_pre_ds_1), 1)
        out = self.decoder_conv_id_1(out_post_up_1)
        out_for_vis, m = self.decoder_deconv1_2(out, m)
        out, m = self.decoder_deconv1_1(out_for_vis, m)

        return out, out_for_vis