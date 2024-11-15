import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from os.path import join

from model.CNN2Layer import *

from model.TCN1D import TCN_IV_1D
from model.tcn import TCN_IV_1D_C

from model.M2M_LSTM import LSTM_MM, GRU_MM, CNN_GRU_MM
from model.RNN_CNN import inverse_model
from model.Unet_1D import Unet_1D

from model.CNN_RNN_Linear import CNN_GRU_Linear

from setting import *
from utils.utils import *
from utils.datasets import SeismicDataset1D
from torch.utils.data import DataLoader

from sklearn.metrics import r2_score
import errno
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = "1" # 2、解决RuntimeError: CUDA error: unspecified launch failure
torch.cuda.device_count()

##------1维模型的选择-----------
model_name = TCN1D_train_p['model_name']
data_flag = TCN1D_train_p['data_flag']
get_F = TCN1D_train_p['get_F']
F = TCN1D_train_p['F']  # 地震数据的处理方式

if model_name == 'TCN':
	choice_model = TCN_IV_1D        # TCN
if model_name == 'tcnc':
	choice_model = TCN_IV_1D_C      #  TCN

if model_name == 'CNN':
	choice_model = CNN              # 2*[conv_1D(1*80) + ReLU]
if model_name == 'CNN_R':
	choice_model = CNN_R            # 2*[conv_1D(1*80) + ReLU] + conv_1D(1*1)
if model_name == 'VishalNet':
	choice_model = VishalNet        #  Conv_1d(1*81) + ReLU + Conv_1d(1*301),
if model_name == 'VishalNet_par':
	choice_model = VishalNet_par    # 

if model_name == 'LSTM_MM':
	choice_model = LSTM_MM          # LSTM模型: 2层LSTM(dropout=0.75) + Linear(特征维度)
if model_name == 'GRU_MM':
	choice_model = GRU_MM           #  GRU模型： 2层GRU(dropout=0.75) + Linear(特征维度)
if model_name == 'CNN_GRU_MM':
	choice_model = CNN_GRU_MM       # CNN-GRU:  1层CNN模型 + 2层GRU(dropout=0.75) + Linear(特征维度)
if model_name == 'CNNN_GRUN':
	choice_model = inverse_model    

if model_name == 'Unet_1D':
	choice_model = Unet_1D          # ，unet模型的1D样式

if model_name == 'CNN_GRU_Linear':
	choice_model = CNN_GRU_Linear   # 

def get_data(no_wells=10, data_flag='SEAM', get_F=get_F):
	'''
		no_well: 均匀采样的井数，
		data_flag: 三个数据集在深度域-时间域是完全对齐的。
			完整剖面数据： 'SEAM', "M2", 
			随机体数据,没有剖面特征： 'Volve', 
		输出的数据格式： (num_traces, channel, depth_samples)
	'''
	if data_flag == 'SEAM':
		# SEAM: 地震波形(1002, 1, 751) ==> (501, 701)
		seismic = np.load(join('data',data_flag,'poststack_seam_seismic.npy')).squeeze()[:, 50:]
		seismic = seismic[::2, :]

		# SEAM: 波阻抗 （1502，3，1501） ==> (501, 701)
		model = np.load(join('data',data_flag,'seam_elastic_model.npy'))[::3,:,::2][:, :, 50:]
		model = model[:,0,:] * model[:,2,:]

		print('original long: ', model.shape, seismic.shape)
		print('model_Mean: ', model.mean(), 'seismic_mean: ', seismic.mean())  # 8800.779; -0.1263349

	if data_flag == 'M2':
		# M2: 地震波形(1, 2721, 701) ==> (2721, 701)
		seismic = np.load(join('data',data_flag,'marmousi_synthetic_seismic.npy')).squeeze()

		# M2: 波阻抗(1, 13601, 2801) ==> (2721, 701)
		model= np.load(join('data',data_flag, 'marmousi_Ip_model.npy')).squeeze()[::5, ::4]

		print('original long: ', model.shape, seismic.shape)
		print('model_Mean: ', model.mean(), 'seismic_mean: ', seismic.mean())  # 5829.911; -0.0010833213
	
	if data_flag == 'M2_F':
		# M2: 地震波形(1, 2721, 701) ==> (2721, 701)
		if F == 'Kirchhoff_PreSDM':
			seismic = np.load(join('data',data_flag,'M2_Kirchhoff_PreSDM.npy')).squeeze()
		if F == 'NMOstack':
			seismic = np.load(join('data',data_flag,'M2_NMOstack.npy')).squeeze()
		if F == 'SYNTHETIC':
			seismic = np.load(join('data',data_flag,'M2_SYNTHETIC.npy')).squeeze()
		if  F == 'WE_PreSDM':
			seismic = np.load(join('data',data_flag,'M2_WE_PreSDM.npy')).squeeze()

		# M2: 波阻抗(1, 13601, 2801) ==> (2721, 701)
		model= np.load(join('data',data_flag, 'marmousi_Ip_model.npy')).squeeze()[::5, ::4]
		seismic = seismic[400:2321,:]
		model = model[400:2321,:]

		print('original long: ', model.shape, seismic.shape)
		print('model_Mean: ', model.mean(), 'seismic_mean: ', seismic.mean())  # 5829.911; -0.0010833213

	if data_flag == 'X70':
		# 地震波形(138, 96)
		seismic = np.load(join('data',data_flag,'Seismic.npy')).squeeze()

		# GR (138, 96)
		model= np.load(join('data',data_flag, 'GR.npy')).squeeze()

		print('original long: ', model.shape, seismic.shape)
		print('model_Mean: ', model.mean(), 'seismic_mean: ', seismic.mean())

	if data_flag == 'Volve':
		'''
			数据集是使用贯序高斯模拟生成的
			Ip: (22801, 160)   mean: 10.191281291523946
			seismic: (22801, 160)    mean: 0.500187737545036
		'''
		seismic = np.load('data/Volve/Volve_synthetic_seismic_w.npy')
		model = np.load('data/Volve/Volve_synthetic_IP_w.npy')
		print('original long: ', model.shape, seismic.shape)
		print('model_Mean: ', model.mean(), 'seismic_mean: ', seismic.mean())  # 10.19; 0.50018

	seismic, model = standardize(seismic, model, no_wells)

	s_L = seismic.shape[-1]
	n = int((s_L//8)*8)
	seismic = seismic[:,:n]
	model = model[:, :n]

	# 2、增加地震数据的频率特征和动态特征。(num_traces, depth_samples) --> (num_traces, depth_samples, 5)
	if get_F:
		seismic = seismic[1:2720, :]
		seismic_feature = n_s_features(seismic, get_F//2)
		seismic = np.transpose(seismic_feature, (0,2,1))  # (num_traces, 5, depth_samples)
		model = model[1:2720, np.newaxis, :]                   # (num_traces, 1, depth_samples)
	else:
		# (num_traces, 1, depth_samples)
		seismic = seismic[:, np.newaxis, :]
		model = model[:, np.newaxis, :]

	print('train long: ',model.shape, seismic.shape)

	return seismic, model

def train(TCN1D_train_p):
	
	# 获取全部的数据，(num_traces, depth_samples)
	seismic, model = get_data(no_wells=TCN1D_train_p['no_wells'], data_flag=TCN1D_train_p['data_flag'])
	print(seismic.shape, model.shape)

	traces_train = np.linspace(0, len(model)-1, TCN1D_train_p['no_wells'], dtype=int)
	train_dataset = SeismicDataset1D(seismic, model, traces_train)
	train_loader = DataLoader(train_dataset, batch_size=TCN1D_train_p['batch_size'])  

	traces_validation = np.linspace(0, len(model)-1, 3, dtype=int)
	val_dataset = SeismicDataset1D(seismic, model, traces_validation)
	val_loader = DataLoader(val_dataset, batch_size=len(val_dataset))
	
	# define device for training
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	# set up models
	model = choice_model(input_dim=get_F+1).to(device)
	
	# Set up loss
	criterion1 = torch.nn.MSELoss()
	criterion2 = torch.nn.L1Loss()  # MAE损失，相对于平方的MSE损失，MAE更线性。
	
	optimizer = torch.optim.Adam(model.parameters(), weight_decay=0.0001, lr=TCN1D_train_p['lr'])
	
	train_loss = []
	val_loss = []
	for epoch in range(TCN1D_train_p['epochs']):

		model.train()
		optimizer.zero_grad()

		for x,y in train_loader:
			# (12, 1, 701)
			y_pred = model(x)

			loss_train1 = criterion1(y_pred, y)  # MSE
			loss_train2 = criterion2(y_pred, y)  # MAE
			loss_train = loss_train1 + loss_train2

			loss_train.backward()
			optimizer.step()
			train_loss.append(loss_train.item())

		for x, y in val_loader:
			# (3, 1, 701)
			model.eval()
			y_pred = model(x)
			loss_val = criterion1(y_pred, y)

			val_loss.append(loss_val.item())

		print('Epoch: {} | Train Loss: {:0.4f} | Val Loss: {:0.4f} \
			'.format(epoch, loss_train.item(), loss_val.item()))

	# save trained models
	if not os.path.isdir('save_train_model'):  # check if directory for saved models exists
		os.mkdir('save_train_model')
	print('train_model_name: %s, train_data_name: %s'%(model_name, data_flag))
	torch.save(model, 'save_train_model/%s_%s.pth'%(model_name, data_flag))

	plt.plot(train_loss,'r')
	plt.plot(val_loss,'k')
	plt.savefig('results/%s_%s.png'%(model_name, data_flag))



if __name__ == '__main__':
	train(TCN1D_train_p=TCN1D_train_p)

