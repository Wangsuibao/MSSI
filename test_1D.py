import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, MaxNLocator
from matplotlib.colors import LinearSegmentedColormap
import os
import torch
from os.path import join

from model.TCN1D import TCN_IV_1D
from model.tcn import TCN_IV_1D_C
from model.CNN2Layer import *
from model.M2M_LSTM import LSTM_MM

from setting import *
from utils.utils import *
from utils.datasets import SeismicDataset1D
from torch.utils.data import DataLoader

from sklearn.metrics import r2_score
from scipy.stats import pearsonr
from skimage.metrics import structural_similarity as ssim
from cv2 import PSNR

import errno
import argparse

##------1维模型的选择-----------
model_name = TCN1D_test_p['model_name']
data_flag = TCN1D_test_p['data_flag']
get_F = TCN1D_train_p['get_F']
F = TCN1D_train_p['F']  # 地震数据的处理方式

def get_data(no_wells=10, data_flag='SEAM', get_F=get_F):
	'''
		no_well: 均匀采样的井数，
		data_flag: 三个数据集在深度域-时间域是完全对齐的。
			完整剖面数据： 'SEAM', "M2", 
			随机体数据,没有剖面特征： 'Volve', 
		输出的数据格式： (num_traces, depth_samples)
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
	# 数据维度： （num_traces, channel=1或5, depth_samples）
	return seismic, model

def show_M2(AI_act,AI_pred,vmin,vmax):
	'''
		绘图，1、高分辨率图像，成图时设置dpi=400,默认为100，表是每英寸400个点
		set_aspect(60/30)  # 在设定extent的情况下设置比例，用于拉伸-压缩纵横比
	'''
	depth_index = np.array(range(5, AI_act.shape[1]+5, 1))*5
	Trace_index = np.array(range(0,AI_act.shape[0]))*6.25

	# plt.imshow(AI_pred.T)
	# plt.show()

	# 一、预测M2剖面
	fig, ax1 = plt.subplots(figsize=(16,8), dpi=400)  # 图像的大小，像素
	ax1.imshow(AI_pred.T, vmin=vmin, vmax=vmax, extent=(0,17000,3500,0))  # (---完整M2-------)

	ax1.xaxis.set_major_locator(MultipleLocator(2000))  # 设置x轴刻度间隔为2000
	ax1.yaxis.set_major_locator(MultipleLocator(1000))  # 设置y轴刻度间隔为1000
	ax1.tick_params(axis='both', labelsize=18)

	ax1.set_aspect(60/30)  # 在设定extent的情况下设置比例，用于拉伸-压缩纵横比
	ax1.set_xlabel('Distance (m)', fontsize=20)
	ax1.set_ylabel('Depth (m)', fontsize=20)
	ax1.set_title('Inversion Profile', fontsize=20)
	plt.savefig('results/%s_%s_test_Pred.png'%(model_name, data_flag))
	plt.close()

	# 二、真实M2剖面
	fig, ax2 = plt.subplots(figsize=(16,8), dpi=400)
	ax2.imshow(AI_act.T, vmin=vmin, vmax=vmax, extent=(0,17000,3500,0))   # (---完整M2-------)

	ax2.xaxis.set_major_locator(MultipleLocator(2000))  # 设置x轴刻度间隔为2000
	ax2.yaxis.set_major_locator(MultipleLocator(1000))  # 设置y轴刻度间隔为1000
	ax2.tick_params(axis='both', labelsize=18)

	ax2.set_aspect(60/30)
	ax2.set_xlabel('Distance (m)', fontsize=20)
	ax2.set_ylabel('Depth (m)', fontsize=20)
	ax2.set_title('Ground-Truth', fontsize=20)
	plt.savefig('results/%s_%s_test_True.png'%(model_name, data_flag))
	plt.close()


	# 三、具体道可视化
	'''
		可视乎道1660和480的原因是：
			1、0-17000米的地震道数据，480道对应透镜状河道，1660对应断层位置背斜
			2、2500-14500米有效地震范围，80和1260
	'''

	B_T = 1660   # 断层位置背斜顶
	H_T = 480    # 透镜状河道

	fig, ax = plt.subplots(figsize=(16,6), dpi=400)  # dpi表示每英寸点
	ax.plot(depth_index, AI_pred.T[:, B_T], linestyle='--', label='Pred', color='blue', linewidth=3.0)  # 1660
	ax.plot(depth_index, AI_act.T[:, B_T], linestyle='-', label='True', color='red', linewidth=3.0)

	ax.xaxis.set_major_locator(MultipleLocator(1000))  # 设置x轴刻度间隔为2000
	ax.yaxis.set_major_locator(MultipleLocator(0.5))  # 设置y轴刻度间隔为1000
	ax.tick_params(axis='both', labelsize=18)

	ax.set_xlabel("Depth(m)", fontsize=20)
	ax.set_ylabel('Impedance', fontsize=20)
	ax.set_title('Trace %s'%B_T, fontsize=20)
	plt.legend(loc='upper left')
	plt.savefig('results/%s_%s_T%s_test.png'%(model_name, data_flag, B_T))
	plt.close()

	fig, ax = plt.subplots(figsize=(16,6), dpi=400)  # dpi表示每英寸点
	ax.plot(depth_index, AI_pred.T[:, H_T], linestyle='--', label='Pred', color='blue', linewidth=3.0)
	ax.plot(depth_index, AI_act.T[:, H_T],  linestyle='-', label='True', color='red', linewidth=3.0)

	ax.xaxis.set_major_locator(MultipleLocator(1000))  # 设置x轴刻度间隔为2000
	ax.yaxis.set_major_locator(MultipleLocator(0.5))  # 设置y轴刻度间隔为1000
	ax.tick_params(axis='both', labelsize=18)

	ax.set_xlabel("Depth(m)", fontsize=20)
	ax.set_ylabel('Impedance', fontsize=20)
	ax.set_title('Trace %s'%H_T, fontsize=20)
	plt.legend(loc='upper left')
	plt.savefig('results/%s_%s_T%s_test.png'%(model_name, data_flag, H_T))
	plt.close()

def show_M2_F(AI_act,AI_pred,vmin,vmax):
	'''
		绘图，1、高分辨率图像，成图时设置dpi=400,默认为100，表是每英寸400个点
		set_aspect(60/30)  # 在设定extent的情况下设置比例，用于拉伸-压缩纵横比
	'''
	depth_index = np.array(range(5, AI_act.shape[1]+5, 1))*5
	Trace_index = np.array(range(0,AI_act.shape[0]))*6.25

	# plt.imshow(AI_pred.T)
	# plt.show()

	# 一、预测M2剖面
	fig, ax1 = plt.subplots(figsize=(16,8), dpi=400)  # 图像的大小，像素
	ax1.imshow(AI_pred.T, vmin=vmin, vmax=vmax, extent=(2500,14500,3500,0))  # (---采集处理地震有效M2-------)

	ax1.xaxis.set_major_locator(MultipleLocator(2000))  # 设置x轴刻度间隔为2000
	ax1.yaxis.set_major_locator(MultipleLocator(1000))  # 设置y轴刻度间隔为1000
	ax1.tick_params(axis='both', labelsize=18)

	ax1.set_aspect(60/30)  # 在设定extent的情况下设置比例，用于拉伸-压缩纵横比
	ax1.set_xlabel('Distance (m)', fontsize=20)
	ax1.set_ylabel('Depth (m)', fontsize=20)
	ax1.set_title('Inversion Profile', fontsize=20)
	plt.savefig('results/%s_%s_test_Pred.png'%(model_name, data_flag))
	plt.close()

	# 二、真实M2剖面
	fig, ax2 = plt.subplots(figsize=(16,8), dpi=400)
	ax2.imshow(AI_act.T, vmin=vmin, vmax=vmax, extent=(2500,14500,3500,0))   # (---采集处理地震有效M2-------)

	ax2.xaxis.set_major_locator(MultipleLocator(2000))  # 设置x轴刻度间隔为2000
	ax2.yaxis.set_major_locator(MultipleLocator(1000))  # 设置y轴刻度间隔为1000
	ax2.tick_params(axis='both', labelsize=18)

	ax2.set_aspect(60/30)
	ax2.set_xlabel('Distance (m)', fontsize=20)
	ax2.set_ylabel('Depth (m)', fontsize=20)
	ax2.set_title('Ground-Truth', fontsize=20)
	plt.savefig('results/%s_%s_test_True.png'%(model_name, data_flag))
	plt.close()


	# 三、具体道可视化
	'''
		可视乎道1660和480的原因是：
			1、0-17000米的地震道数据，480道对应透镜状河道，1660对应断层位置背斜
			2、2500-14500米有效地震范围，80和1260
	'''
	B_T = 1260   # 断层位置背斜顶
	H_T = 80     # 透镜状河道

	fig, ax = plt.subplots(figsize=(16,6), dpi=400)  # dpi表示每英寸点
	ax.plot(depth_index, AI_pred.T[:, B_T], linestyle='--', label='Pred', color='blue', linewidth=3.0)  # 1660
	ax.plot(depth_index, AI_act.T[:, B_T], linestyle='-', label='True', color='red', linewidth=3.0)

	ax.xaxis.set_major_locator(MultipleLocator(1000))  # 设置x轴刻度间隔为2000
	ax.yaxis.set_major_locator(MultipleLocator(0.5))  # 设置y轴刻度间隔为1000
	ax.tick_params(axis='both', labelsize=18)

	ax.set_xlabel("Depth(m)", fontsize=20)
	ax.set_ylabel('Impedance', fontsize=20)
	ax.set_title('Trace %s'%B_T, fontsize=20)
	plt.legend(loc='upper left')
	plt.savefig('results/%s_%s_T%s_test.png'%(model_name, data_flag, B_T))
	plt.close()

	fig, ax = plt.subplots(figsize=(16,6), dpi=400)  # dpi表示每英寸点
	ax.plot(depth_index, AI_pred.T[:, H_T], linestyle='--', label='Pred', color='blue', linewidth=3.0)
	ax.plot(depth_index, AI_act.T[:, H_T],  linestyle='-', label='True', color='red', linewidth=3.0)

	ax.xaxis.set_major_locator(MultipleLocator(1000))  # 设置x轴刻度间隔为2000
	ax.yaxis.set_major_locator(MultipleLocator(0.5))  # 设置y轴刻度间隔为1000
	ax.tick_params(axis='both', labelsize=18)

	ax.set_xlabel("Depth(m)", fontsize=20)
	ax.set_ylabel('Impedance', fontsize=20)
	ax.set_title('Trace %s'%H_T, fontsize=20)
	plt.legend(loc='upper left')
	plt.savefig('results/%s_%s_T%s_test.png'%(model_name, data_flag, H_T))
	plt.close()

def show_x70(seismic,AI_act,AI_pred,vmin,vmax):
	'''
		绘图，1、高分辨率图像，成图时设置dpi=400, 默认为100，表是每英寸400个点
		set_aspect(60/30)  # 在设定extent的情况下设置比例，用于拉伸-压缩纵横比
		X70，数据维度（Trance，time）（138,96）真实长度：long=1492m，T=48ms,
	'''
	depth_index = np.array(range(0, AI_act.shape[1], 1))*0.5

	# 自定义GR表示岩性的颜色
	colors = [(1, 0, 0), (1, 1, 1), (0, 0, 1)]   # 红色到蓝色的渐变
	n_bins = 100  # 设定渐变的分段数
	cmap_name = 'my_custom_cmap'
	cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)

	# 一、预测X70剖面  T=48ms, long=1492m
	fig, ax1 = plt.subplots(figsize=(16,8), dpi=100)  # 图像的大小，像素
	ax1.imshow(AI_pred.T, vmin=vmin, vmax=vmax, extent=(0,1492,48,0), aspect=10, interpolation='bicubic', cmap=cm)

	ax1.xaxis.set_major_locator(MultipleLocator(200))  # 设置x轴刻度间隔为200
	ax1.yaxis.set_major_locator(MultipleLocator(20))  # 设置y轴刻度间隔为20
	ax1.tick_params(axis='both', labelsize=18)

	# ax1.set_aspect(60/30)  # 在设定extent的情况下设置比例，用于拉伸-压缩纵横比,不能和imshow中的aspect=10重复
	ax1.set_xlabel('Distance(m)', fontsize=20)
	ax1.set_ylabel('Time(ms)', fontsize=20)
	ax1.set_title('Inversion Profile', fontsize=20)
	plt.savefig('results/%s_%s_test_Pred.png'%(model_name, data_flag))
	plt.close()

	# 二、真实X70剖面   T=48ms, long=1492m
	fig, ax2 = plt.subplots(figsize=(16,8), dpi=100)
	ax2.imshow(AI_act.T, vmin=vmin, vmax=vmax, extent=(0,1492,48,0), aspect=10, interpolation='bicubic', cmap=cm)

	ax2.xaxis.set_major_locator(MultipleLocator(200))
	ax2.yaxis.set_major_locator(MultipleLocator(20))
	ax2.tick_params(axis='both', labelsize=18)

	ax2.set_xlabel('Distance(m)', fontsize=20)
	ax2.set_ylabel('Time(ms)', fontsize=20)
	ax2.set_title('Ground-Truth', fontsize=20)
	plt.savefig('results/%s_%s_test_True.png'%(model_name, data_flag))
	plt.close()

	# 三、具体道可视化井索引：4, 21, 42, 64, 89, 113, 131
	# 需要归一化显示（-2.0，1.5）

	B_T = 64
	# 预测+真实，归一化
	fig, ax = plt.subplots(figsize=(16,6), dpi=400)  # dpi表示每英寸点
	ax.plot(depth_index, nor_1d(AI_pred.T[:, B_T]), linestyle='--', label='Pred', color='blue', linewidth=3.0)  # 1660
	ax.plot(depth_index, nor_1d(AI_act.T[:, B_T]), linestyle='-', label='True', color='red', linewidth=3.0)

	ax.xaxis.set_major_locator(MultipleLocator(20))  # 设置x轴刻度间隔为2000
	ax.yaxis.set_major_locator(MultipleLocator(0.5))  # 设置y轴刻度间隔为1000
	ax.tick_params(axis='both', labelsize=18)

	ax.set_xlabel("Time(ms)", fontsize=20)
	ax.set_ylabel('Pseudo-GR', fontsize=20)
	ax.set_title('Trace %s'%B_T, fontsize=20)
	plt.legend(loc='upper left')
	plt.savefig('results/%s_%s_T%s_test_nor.png'%(model_name, data_flag,B_T))
	plt.close()

	# 预测+真实，非归一化，标准化导致
	fig2, ax2 = plt.subplots(figsize=(16,6), dpi=400)  # dpi表示每英寸点
	ax2.plot(depth_index, AI_pred.T[:, B_T], linestyle='--', label='Pred', color='blue', linewidth=3.0)
	ax2.plot(depth_index, AI_act.T[:, B_T],  linestyle='-', label='True', color='red', linewidth=3.0)

	ax2.xaxis.set_major_locator(MultipleLocator(20))  # 设置x轴刻度间隔为200
	ax2.yaxis.set_major_locator(MultipleLocator(0.5))  # 设置y轴刻度间隔为0.5
	ax2.tick_params(axis='both', labelsize=18)

	ax2.set_xlabel("Time(m)", fontsize=20)
	ax2.set_ylabel('Pseudo-GR', fontsize=20)
	ax2.set_title('Trace %s'%B_T, fontsize=20)
	plt.legend(loc='upper left')
	plt.savefig('results/%s_%s_T%s_test.png'%(model_name, data_flag,B_T))
	plt.close()



def test(TCN1D_test_p):
	"""Function tests the trained network on SEAM and Marmousi sections and 
	prints out the results"""
	
	# 获取全部数据
	seismic, model = get_data(TCN1D_test_p['no_wells'], TCN1D_test_p['data_flag'])
																		  
	# define device for training
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	
	# 全部的数据用于最后测试，可视化 
	traces_test = np.arange(len(model), dtype=int)
	
	test_dataset = SeismicDataset1D(seismic, model, traces_test)
	test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)  # 按顺序逐个读取
	
	# 获取模型
	if not os.path.isdir('save_train_model/'):
		raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), 'save_train_model/')
	print('test_model_name: %s, test_data_name: %s'%(model_name, data_flag))
	inver_model = torch.load('save_train_model/%s_%s.pth'%(model_name, data_flag)).to(device)
	
	# infer on SEAM
	print("\nInferring ...")
	x, y = test_dataset[0]  # get a sample
	AI_pred = torch.zeros((len(test_dataset), y.shape[-1])).float().to(device)
	AI_act = torch.zeros((len(test_dataset), y.shape[-1])).float().to(device)
	
	mem = 0
	with torch.no_grad():
		inver_model.eval()
		for i, (x,y) in enumerate(test_loader):
			y_pred  = inver_model(x)
			AI_pred[mem:mem+len(x)] = y_pred.squeeze().data
			AI_act[mem:mem+len(x)] = y.squeeze().data
			mem += len(x)
			# del x, y, y_pred
	
	vmin, vmax = AI_act.min(), AI_act.max()

	# 真实数据和预测数据的不同类型的统计误差，包含：
	AI_pred = AI_pred.detach().cpu().numpy()
	AI_act = AI_act.detach().cpu().numpy()

	# R2_score(判别系数), PCC(皮尔逊相关系数), SSIM(结构相似性指数)， PSNR(峰值信噪比)， MSE
	print('r^2 score: {:0.4f}'.format(r2_score(AI_act.T, AI_pred.T)))  # 相似度
	pcc, _ = pearsonr(AI_act.T.ravel(), AI_pred.T.ravel())             # 相似性，相关性[-1,1]
	print('PCC: {:0.4f}'.format(pcc))

	print('ssim: {:0.4f}'.format(ssim(AI_act.T, AI_pred.T)))           # 结构相似度[-1,1]
	print('PSNR: {:0.4f}'.format(PSNR(AI_act.T, AI_pred.T)))           # 分贝（dB）单位，值越高图像质量越好（最大值361）

	print('MSE: {:0.4f}'.format(np.sum((AI_pred-AI_act).ravel()**2)/AI_pred.size))
	print('MAE: {:0.4f}'.format(np.sum(np.abs(AI_pred - AI_act)/AI_pred.size)))
	print('MedAE: {:0.4f}'.format(np.median(np.abs(AI_pred - AI_act))))

	# 保存可是化结果
	if data_flag == 'M2': 
		show_M2(AI_act,AI_pred,vmin,vmax)
	if data_flag == 'M2_F': 
		show_M2_F(AI_act,AI_pred,vmin,vmax)

	if data_flag == 'X70': 
		show_x70(seismic, AI_act, AI_pred, vmin, vmax)



if __name__ == '__main__':
	# get_data(no_wells=10, data_flag='M2')
	test(TCN1D_test_p)