'''
	数据获取; seismic, model(地球物理参数)的数据格式是ndarray： (num_traces, channel, depth_samples) 等长
	数据样式是矩形。
'''

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import numpy as np

class SeismicDataset1D(Dataset):
	"""
		训练1D model 时的数据组织方式
		seismic, model的数据格式是： (num_traces, channel, depth_samples) 等长，矩形数据。
		返回：（channel, depth_samples）
	"""
	def __init__(self, seismic, model, trace_indices):
		self.seismic = seismic
		self.model = model
		self.trace_indices = trace_indices
	
		assert min(trace_indices) >= 0 and max(trace_indices) + 1 <= len(seismic),"Seismic patch accessing traces out of geometry of the data!"
	
	def __getitem__(self, index):
		# 随机获取index,提取数据，返回格式tensor, 这里返回只考虑（channel，depth_samples）,batch会自动添加
		# DataLoader自动循环，每次随机选取一个样本，最后形成一个batch
		trace_index = self.trace_indices[index]
		
		x = torch.tensor(self.seismic[trace_index, :, :], dtype=torch.float).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
		y = torch.tensor(self.model[trace_index, :, :], dtype=torch.float).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
		return x, y

	def __len__(self):
		return len(self.trace_indices)




class UnsupervisedSeismicDataset(Dataset):
	def __init__(self, seismic, trace_indices):
		self.seismic = seismic
		self.trace_indices = trace_indices
	
		assert min(trace_indices) >= 0 and max(trace_indices) + 1 <= len(seismic),"Seismic patch accessing traces out of geometry of the data!"
	
	def __getitem__(self, index):
		trace_index = self.trace_indices[index]
		x = torch.tensor(self.seismic[trace_index][np.newaxis, :], dtype=torch.float).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

		return x

	def __len__(self):
		return len(self.trace_indices) 


class SeismicDataset2D(Dataset):
	"""Dataset class for loading 2D seismic images for 2-D TCN"""
	def __init__(self, seismic, model, trace_indices, width):
		self.seismic = seismic
		self.model = model
		self.trace_indices = trace_indices
		self.width = width

		assert min(trace_indices) - int(width/2) >= 0 and max(trace_indices) + int(width/2) + 1 <= len(seismic),"Seismic patch accessing traces out of geometry of the data!"

	def __getitem__(self, index):
		offset = int(self.width/2)
		trace_index = self.trace_indices[index]
		x = torch.tensor(self.seismic[trace_index-offset:trace_index+offset+1].T[np.newaxis, :, :], dtype=torch.float).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
		y = torch.tensor(self.model[trace_index][np.newaxis, :], dtype=torch.float).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
		return x, y

	def __len__(self):
		return len(self.trace_indices)

