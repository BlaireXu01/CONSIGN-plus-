import torch
import matplotlib.pyplot as plt
import torch.utils.checkpoint as checkpoint
from helpers import dice_coeff
import torch.nn as nn
import torch.nn.functional as F
from config import Config

class EncoderBlock(nn.Module):
    def __init__(self, in_size, out_size,p_):
        super(EncoderBlock, self).__init__()
        block = []

        block.append(nn.Conv2d(in_size, out_size, kernel_size=3, padding=1))
        block.append(nn.ReLU())
        block.append(nn.BatchNorm2d(out_size))

        block.append(nn.Dropout(p=p_))

        block.append(nn.Conv2d(out_size, out_size, kernel_size=3, padding=1))
        block.append(nn.ReLU())
        block.append(nn.BatchNorm2d(out_size))

        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out

class EncoderBlock_decoder(nn.Module):
    def __init__(self, in_size, out_size,p_):
        super(EncoderBlock_decoder, self).__init__()
        block = []
        
        block.append(nn.Conv2d(in_size, out_size, kernel_size=3, padding=1))
        block.append(nn.ReLU())
        block.append(nn.BatchNorm2d(out_size))

        block.append(nn.Dropout(p=p_))
        
        block.append(nn.Conv2d(out_size, out_size, kernel_size=3, padding=1))
        block.append(nn.ReLU())
        block.append(nn.BatchNorm2d(out_size))

        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out

class MRIEncoder(nn.Module):
    """
    Class for Image encoder
    """
    def __init__(self, depth, flt):
      super(MRIEncoder, self).__init__()
      self.depth = depth
      self.flt = flt
      self.down_path = nn.ModuleList()
      prev_channels = 1
      for i in range(depth):
          self.down_path.append(
              EncoderBlock(prev_channels, flt * 2 ** i, Config.dropout*(i!=4))
          )
          prev_channels = flt * 2 ** i

    def forward(self, x, mode):

      skip = {}
      for i, down in enumerate(self.down_path):

          x = down(x)
          skip[''+str(i)+''] = x

          if i != len(self.down_path) - 1:
              x = F.max_pool2d(x, 2)

      return skip


class Decoder(nn.Module):
    def __init__(self, in_size, nr_labels):
      super(Decoder, self).__init__()

      self.up1 = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2),
                nn.Conv2d(in_size, in_size//2, kernel_size=1),)

      self.conv1 = EncoderBlock_decoder(in_size, in_size//2,Config.dropout)

      self.up2 = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2),
                nn.Conv2d(in_size//2, in_size//4, kernel_size=1),)

      self.conv2 = EncoderBlock_decoder(in_size//2, in_size//4,Config.dropout)

      self.up3 = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2),
                nn.Conv2d(in_size//4, in_size//8, kernel_size=1),)

      self.conv3 = EncoderBlock_decoder(in_size//4, in_size//8,Config.dropout)

      self.up4 = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2),
                nn.Conv2d(in_size//8, in_size//16, kernel_size=1),)

      self.conv4 = EncoderBlock_decoder(in_size//8, in_size//16,Config.dropout)
      self.final = nn.Conv2d(in_size//16, nr_labels, kernel_size=1)

    def forward(self,x):
        # upsample
        up1 = self.up1(x['4'])
        out1 = torch.cat([up1, x['3']], 1)
        out1 = self.conv1(out1)

        # upsample
        up2 = self.up2(out1)
        out2 = torch.cat([up2, x['2']], 1)
        out2 = self.conv2(out2)

        # upsample
        up3 = self.up3(out2)
        out3 = torch.cat([up3, x['1']], 1)
        out3 = self.conv3(out3)

        # upsample
        up4 = self.up4(out3)
        out4 = torch.cat([up4, x['0']], 1)
        out4 = self.conv4(out4)
        final = self.final(out4)

        return final

class Conformal_UNet(nn.Module):
    def __init__(self, image_encoder, upsampler, loss, reason):
        super().__init__()
        self.image_encoder = image_encoder
        self.upsampler = upsampler
        self.loss = loss
        self._reason = reason

    def forward(self, query, is_val, epoch=0):
        """
        Args:
        """
        # extract query image and segmenation
        image, seg = query

        encoded_image = self.image_encoder(image, 'query')
        pred = self.upsampler(encoded_image)

        if self._reason == 'train' and not is_val:
            return self.loss(pred, seg[:,0,:,:])
        elif self._reason == 'train' and is_val:
            return dice_coeff(seg, pred)
        else:
            return pred
