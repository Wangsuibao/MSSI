
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


class TemporalBlock(nn.Module):

	def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
		'''
			in_channel, out_channel, kernel_size, stride, dilation, padding
			空洞卷积是，扩大卷积核，稀疏卷积 

			下面设置保证输入输出长度一致
			kernel_size, stride=1, dilation=dialation_size, padding=(kernel_size-1)/2*dialation_size
		'''
		super(TemporalBlock, self).__init__()

		self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
		self.relu1 = nn.ReLU()
		self.dropout1 = nn.Dropout(dropout)

		self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
		self.relu2 = nn.ReLU()
		self.dropout2 = nn.Dropout(dropout)

		self.net = nn.Sequential(self.conv1, self.relu1, self.dropout1, 
								self.conv2, self.relu2, self.dropout2)
		# 处理残差连接,把输出处理成输出一样的channel, kernel=1
		self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
		self.relu = nn.ReLU()
		self.init_weights()

	def init_weights(self):
		self.conv1.weight.data.normal_(0,0.01)
		self.conv2.weight.data.normal_(0,0.01)
		if self.downsample is not None:
			self.downsample.weight.data.normal_(0,0.01)

	def forward(self, x):
		out = self.net(x)
		res = x if self.downsample is None else self.downsample(x)
		return self.relu(out + res)  #============================3、残差连接===================

class TCN(nn.Module):
	def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
		'''
			num_input: 输出channel
			num_channels: [3,6,6,6,5], 每一值表示下一层的输出;
				in：[1,3,6,6,6]
				out:[3,6,6,6,5]
		'''
		super(TCN, self).__init__()
		layers = []
		num_levels = len(num_channels)

		for i in range(num_levels):
			dialation_size = 2**i
			in_channels = num_inputs if i==0 else num_channels[i-1]
			out_channels = num_channels[i]

			'''
			padding=(kernel_size-1)/2*dialation_size ； 保证kernel_size为奇数
			kernel_size, stride=1, dilation=dialation_size, padding=(kernel_size-1)/2*dialation_size
			保证每个隐层的输出长度一致
			'''
			layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dialation_size, 
				                     padding=int((kernel_size-1)/2*dialation_size), dropout=dropout)]

		self.network = nn.Sequential(*layers)

	def forward(self, x):
		return self.network(x)

# ---------------------------------------------------具体化------------------------------
class TCN_IV_1D(nn.Module):

	def __init__(self):
		super(TCN_IV_1D, self).__init__()

		self.tcn_local = TCN(num_inputs=1, num_channels=[3,6,6,6,6,6,6,6,5], kernel_size=15, dropout=0.2)
		self.regression = nn.Conv1d(in_channels=5, out_channels=1, kernel_size=1)

	def forward(self, input):
		out = self.tcn_local(input)
		out = self.regression(out)
		return out


if __name__ == '__main__':
	model = TCN_IV_1D()
	print(model)

