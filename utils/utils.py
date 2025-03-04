import zipfile
import numpy as np
from scipy import signal
from scipy.fftpack import fft, ifft
from scipy.fftpack import dct
from sklearn.decomposition import PCA

import torch
from os.path import join
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MultipleLocator, MaxNLocator

def extract(source_path, destination_path):
	"""从source_path 给定的zip文件，提取所有文件到destination_path"""
	
	with zipfile.ZipFile(source_path, 'r') as zip_ref:
		zip_ref.extractall(destination_path)
	
	
def standardize(seismic, model, no_wells):
	seismic_normalized = (seismic - seismic.mean())/ seismic.std()
	train_indices = (np.linspace(0, len(model)-1, no_wells, dtype=np.int))  # 均匀采样的道索引
	# 标准化通过索引获取的训练数据，因为没有获取的数据默认是不知道的，所以不能引入均值
	model_normalized = (model - model[train_indices].mean()) / model[train_indices].std() 
	
	return seismic_normalized, model_normalized


def nor_1d(data):
	dmax = data.max()
	dmin = data.min()
	nor_data = (data-dmin)/(dmax-dmin)
	return nor_data

def std_2d(data):
	mu = np.mean(data, axis=0)[np.newaxis,:]
	sigma = np.std(data, axis=0)[np.newaxis,:] 
	std_data = (data - mu) / sigma 
	return std_data

def nor_2d(data):
	dmax = np.max(data, axis=0)[np.newaxis,:]
	dmin = np.min(data, axis=0)[np.newaxis,:]  
	nor_data = (data-dmin)/(dmax-dmin)    
	return nor_data

def trace_F_P_D(s, n_components):
	'''
		完成一个地震道的特征提取,
	'''
	pre_pha = 0.97
	pha_s = np.append(s[0], s[1:] - pre_pha * s[:-1]) 

	M = len(pha_s)
	widths = np.arange(1, int(M//2)+1)
	w = 5
	cwtm = signal.cwt(pha_s, signal.morlet2, widths, w=w)
	cwtm = np.abs(cwtm)         # 0-0.7

	# plt.imshow(cwtm, extent=[1, 701, 1, 350], cmap='PRGn', aspect='auto')  # extent=(左,右,下，上)
	# plt.show()
	cwtm = cwtm.T

	pca = PCA(n_components=n_components)
	pca.fit(cwtm)
	prime_f = pca.transform(cwtm)

	prime_f_1 = np.diff(prime_f, axis=0, prepend=0)
	prime_f_2 = np.diff(prime_f_1, axis=0, prepend=0)  # (n,2)

	s_ = s[:,np.newaxis]
	s_features = np.concatenate((nor_2d(s_), nor_2d(prime_f), nor_2d(prime_f_2)), axis=1)
	# print('seismic features shape (num_line, time, features): ', s_features.shape)

	return s_features


def n_s_features(seismic, n_components):
	s_features_all = []
	for i in range(seismic.shape[0]):
		s = seismic[i,:]
		s_features = trace_F_P_D(s, n_components)  # (701, 5)
		s_features_all.append(s_features)  # 2719个
	s_features_all_ndarray = np.stack(s_features_all, axis=0)  # (2719, 701, 5)
	return s_features_all_ndarray


