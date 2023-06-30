import torch
import torch.nn as nn
import numpy as np

from src.DETCTCNN.model.layers2D import ConvBlock


class EncoderBlock(nn.Module):
	def __init__(self, channels=(64,128,256)):
		super().__init__()
		self.channels = channels
		self.encBlocks = nn.ModuleList(
			[ConvBlock(channels[i], channels[i + 1], use_bn=True)
			 	for i in range(len(channels) - 1)])
		self.pool = nn.MaxPool2d(2)

	def forward(self,x):
		outs = []
		for block in self.encBlocks:
			x = self.pool(x)
			x = block(x)
			outs.append(x)
		return outs 

class DecoderBlock(nn.Module):
	def __init__(self, channels=(256,128,64)):
		super().__init__()
		self.channels = channels
		# initialize the number of channels, upsampler blocks, and
		# decoder blocks
		self.channels = channels
		self.dropout = nn.Dropout()
		self.upconvs = nn.ModuleList(
			[nn.ConvTranspose2d(channels[i], channels[i + 1], 2, 2)
			 	for i in range(len(channels) - 1)])
		self.dec_blocks = nn.ModuleList(
			[ConvBlock(channels[i], channels[i + 1], use_bn=True)
			 	for i in range(len(channels) - 1)])
		# # Missing one conv block
		self.dec_blocks_2 = nn.ModuleList(
			[ConvBlock(channels[i+1], channels[i + 1], use_bn=True)
			 	for i in range(len(channels) - 1)])

	def forward(self, x, connections):
		for i in range(len(self.channels) - 1):

			x = self.upconvs[i](x)
			x = torch.cat([x, connections[i]], dim=1)
			x = self.dec_blocks[i](x)
			x = self.dropout(x)
			x = self.dec_blocks_2[i](x)
		return x

# Basic out channel is a multiplier
class Unet2DMC(nn.Module):
	def __init__(self,input_channels=2,with_1conv=True, use_bn=False, depth=3, basic_out_channel=640, n_labels=7):
		super(Unet2DMC, self).__init__()
		self.with_1conv=with_1conv
		self.input_channels = input_channels
		self.init_conv = nn.Sequential(
    		ConvBlock(input_channels, 40, kernel=(1,1), use_bn=use_bn),
    		ConvBlock(40, 40, kernel=(1,1), use_bn=use_bn)
		)
		self.pre_conv = nn.Sequential(
    		ConvBlock(input_channels, 32, kernel=(1,1), use_bn=use_bn),
    		ConvBlock(32, 64, kernel=(1,1), use_bn=use_bn)
		)
		encoder_channels = basic_out_channel*np.array([2**(i) for i in range(depth)])
		self.encoder = EncoderBlock(channels=encoder_channels) 
		decoder_channels = basic_out_channel*np.array([2**(i) for i in reversed(range(depth))]) 
		self.decoder = DecoderBlock(channels=decoder_channels)
		self.final = ConvBlock(basic_out_channel, n_labels, kernel=(1, 1))

	def forward(self,x):
		out = x
		init = self.init_conv(x)
		
		pre = self.pre_conv(x)
		out = self.encoder(pre)
		connections = list(reversed([pre] + out[::-1][1:] ))
		connections.append(torch.cat([pre, init], dim=1))
		# Get last output for decoder and rest for unet connections
		out = self.decoder(out[-1],connections)
		# Add other residual connections
		out = self.final(out)
		out = nn.functional.softmax(out,dim=1)
		return out




	
