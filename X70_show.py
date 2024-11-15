import numpy as np
import matplotlib.pyplot as plt 
from matplotlib.ticker import MultipleLocator, MaxNLocator
from matplotlib.colors import LinearSegmentedColormap
import torch
from os.path import join

data_flag = 'x70'

def show_x70_S_T_G(data_flag):
	'''
		三个数据是归一化的。
		GR 和 波阻抗 在表示砂岩时，GR是低值，波阻抗是高值
	'''
	# 地震波形(138, 96)
	seismic = np.load(join('data',data_flag,'Seismic.npy')).squeeze()
	# GR (138, 96)
	model= np.load(join('data',data_flag, 'GR.npy')).squeeze()
	# GR (138, 96)
	# model= np.load(join('data',data_flag, 'Impedance.npy')).squeeze()

	print('original long: ', model.shape, seismic.shape)
	print('model_Mean: ', model.mean(), 'seismic_mean: ', seismic.mean())  # 5829.911; -0.0010833213

	# 1、地震剖面   # T=48ms, long=1492m
	fig3, ax3 = plt.subplots(figsize=(16,8), dpi=100)  # 图像的大小，像素,  色系： 'seismic'，'Rainbow'
	ax3.imshow(seismic.T, vmin=0, vmax=1, extent=(0,1492,48,0), aspect=10, interpolation='bilinear', cmap='seismic')
	ax3.xaxis.set_major_locator(MultipleLocator(200))  # 设置x轴刻度间隔为200
	ax3.yaxis.set_major_locator(MultipleLocator(20))  # 设置y轴刻度间隔为20
	ax3.tick_params(axis='both', labelsize=18)

	ax3.set_xlabel('Distance (m)', fontsize=20)
	ax3.set_ylabel('Time (ms)', fontsize=20)
	ax3.set_title('Seismic', fontsize=20)
	plt.savefig('results/%s_%s_show_seismic.png'%('seismic', data_flag))
	plt.close()

	# 2、GR 模型剖面  # T=48ms, long=1492m
	colors = [(1, 0, 0), (1, 1, 1), (0, 0, 1)]   # 红色到蓝色的渐变
	n_bins = 100  # 设定渐变的分段数
	cmap_name = 'my_custom_cmap'
	cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)

	fig3, ax3 = plt.subplots(figsize=(16,8), dpi=100)  # 图像的大小，像素,  色系： 'seismic'，'Rainbow'
	ax3.imshow(model.T, vmin=0, vmax=1, extent=(0,1492,48,0), aspect=10, interpolation='bilinear', cmap=cm)
	ax3.xaxis.set_major_locator(MultipleLocator(200))  # 设置x轴刻度间隔为200
	ax3.yaxis.set_major_locator(MultipleLocator(20))  # 设置y轴刻度间隔为20
	ax3.tick_params(axis='both', labelsize=18)

	ax3.set_xlabel('Distance (m)', fontsize=20)
	ax3.set_ylabel('Time (ms)', fontsize=20)
	ax3.set_title('model', fontsize=20)
	plt.savefig('results/%s_%s_show_GR.png'%('seismic', data_flag))
	plt.close()


	# 3、具体地震道-GR道
	B_T = 64
	depth_index = np.array(range(0, model.shape[1], 1))*0.5
	fig, ax = plt.subplots(figsize=(16,6), dpi=400)  # dpi表示每英寸点
	ax.plot(depth_index, seismic.T[:, B_T], linestyle='--', label='Seismic', color='blue', linewidth=3.0)
	ax.plot(depth_index, model.T[:, B_T], linestyle='-', label='Pseudo-model', color='red', linewidth=3.0)

	ax.xaxis.set_major_locator(MultipleLocator(20))  # 设置x轴刻度间隔为2000
	ax.yaxis.set_major_locator(MultipleLocator(0.5))  # 设置y轴刻度间隔为1000
	ax.tick_params(axis='both', labelsize=18)

	ax.set_xlabel("Time(ms)", fontsize=20)
	ax.set_title('Trace %s'%B_T, fontsize=20)
	plt.legend(loc='upper left')
	plt.savefig('results/%s_%s_T%s_test.png'%('model_Trance', data_flag, B_T))
	plt.close()

show_x70_S_T_G(data_flag)