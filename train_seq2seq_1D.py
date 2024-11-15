import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
from os.path import join

from model.LSTM_seq import seq2seq, Encoder, Decoder
from model.GRU_seq import seq2seq_GRU, Encoder_GRU, Decoder_GRU
from model.GRU_attention_seq import seq2seqAtt, Attention, EncoderAtt, DecoderAtt
from model.MSSI import seq2seqLAS, AttentionLAS, EncoderLAS, DecoderLAS

from setting import *
from utils.utils import *
from utils.datasets import SeismicDataset1D
from torch.utils.data import DataLoader

from sklearn.metrics import r2_score
import errno
import argparse

torch.backends.cudnn.benchmark = False   # 1、解决RuntimeError: cuDNN error: CUDNN_STATUS_INTERNAL_ERROR

os.environ["CUDA_VISIBLE_DEVICES"] = "1" # 2、解决RuntimeError: CUDA error: unspecified launch failure
torch.cuda.device_count()


##------1维模型的选择-----------
model_name = TCN1D_train_p['model_name']
data_flag = TCN1D_train_p['data_flag']
get_F = TCN1D_train_p['get_F']

HID_DIM = 64 


INPUT_DIM = get_F + 1  
OUTPUT_DIM = 1


ENC_EMB_DIM = 8 
DEC_EMB_DIM = 8 

ENC_HID_DIM = 32 
DEC_HID_DIM = 32 

# 基本参数
N_LAYERS = 1  
ENC_DROPOUT = 0.2
DEC_DROPOUT = 0.2


if model_name == 'seq2seq_LSTM':
	choice_model = seq2seq
	enc = Encoder(INPUT_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
	dec = Decoder(OUTPUT_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)

if model_name == 'seq2seq_GRU':
	choice_model = seq2seq_GRU    # seq2seq
	enc = Encoder_GRU(INPUT_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
	dec = Decoder_GRU(OUTPUT_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)

if model_name == 'seq2seqAtt':
	choice_model = seq2seqAtt     # Att + seq2seq
	attn = Attention(ENC_HID_DIM, DEC_HID_DIM)
	enc = EncoderAtt(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT)
	dec = DecoderAtt(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT, attn)

if model_name == 'MSSI':
	choice_model = seq2seqLAS            # MM + Att + seq2seq
	attn = AttentionLAS(ENC_HID_DIM, DEC_HID_DIM)
	enc = EncoderLAS(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT)
	dec = DecoderLAS(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT, attn)	


def init_weights(m):
	'''
		模型初始化
	'''
	for name, param in m.named_parameters():
		nn.init.uniform_(param.data, -0.08, 0.08)
		
def get_data(no_wells=10, data_flag='SEAM', get_F=get_F):
	'''
		no_well: 均匀采样的井数，
		data_flag: 
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

		print(model.shape, seismic.shape)
		print('model_Mean: ', model.mean(), 'seismic_mean: ', seismic.mean())  # 8800.779; -0.1263349

	if data_flag == 'M2':
		# M2: 地震波形(1, 2721, 701) ==> (2721, 701)
		seismic = np.load(join('data',data_flag,'marmousi_synthetic_seismic.npy')).squeeze()

		# M2: 波阻抗(1, 13601, 2801) ==> (2721, 701)
		model= np.load(join('data',data_flag, 'marmousi_Ip_model.npy')).squeeze()[::5, ::4]

		print(model.shape, seismic.shape)
		print('model_Mean: ', model.mean(), 'seismic_mean: ', seismic.mean())  # 5829.911; -0.0010833213
		
	if data_flag == 'Volve':
		'''
			Ip: (22801, 160)   mean: 10.191281291523946
			seismic: (22801, 160)    mean: 0.500187737545036
		'''
		seismic = np.load('data/Volve/Volve_synthetic_seismic_w.npy')
		model = np.load('data/Volve/Volve_synthetic_IP_w.npy')
		print(model.shape, seismic.shape)
		print('model_Mean: ', model.mean(), 'seismic_mean: ', seismic.mean())  # 10.19; 0.50018

	seismic, model = standardize(seismic, model, no_wells)
	s_L = seismic.shape[-1]
	n = int((s_L//8)*8)
	seismic = seismic[:,:n]
	model = model[:, :n]

	if get_F:
		seismic = seismic[1:2720, :]
		seismic_feature = n_s_features(seismic)
		seismic = np.transpose(seismic_feature, (0,2,1))  # (num_traces, 5, depth_samples)
		model = model[1:2720, np.newaxis, :]                   # (num_traces, 1, depth_samples)
	else:
		# (num_traces, 1, depth_samples)
		seismic = seismic[:, np.newaxis, :]
		model = model[:, np.newaxis, :]

	print('train long: ',model.shape, seismic.shape)

	# 数据维度： （num_traces, channel=1或5, depth_samples）
	return seismic, model

# seismic, model = get_data(no_wells=10, data_flag='Volve')

def train(TCN1D_train_p):
	
	seismic, label = get_data(no_wells=TCN1D_train_p['no_wells'], data_flag=TCN1D_train_p['data_flag'])

	traces_train = np.linspace(0, len(label)-1, TCN1D_train_p['no_wells'], dtype=int)
	train_dataset = SeismicDataset1D(seismic, label, traces_train)
	train_loader = DataLoader(train_dataset, batch_size=TCN1D_train_p['batch_size']) 

	traces_validation = np.linspace(0, len(label)-1, 3, dtype=int)
	val_dataset = SeismicDataset1D(seismic, label, traces_validation)
	val_loader = DataLoader(val_dataset, batch_size=len(val_dataset))
	
	# define device for training
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	# device = 'cpu'
	
	# set up models
	model = choice_model(enc, dec, device).to(device)
	model.apply(init_weights)
	
	# Set up loss
	# criterion = torch.nn.MSELoss()
	criterion = torch.nn.L1Loss()  # MAE损失，相对于平方的MSE损失，MAE更线性。
	# criterion = criterion_1 + criterion_2
	
	# weight_decay默认为0，抑制模型过拟合，在损失函数中增加模型权重的L2范数作为惩罚项，防止模型参数太大，减少模型复杂度
	optimizer = torch.optim.Adam(model.parameters(), weight_decay=0.0001, lr=TCN1D_train_p['lr'])
	
	train_loss = []
	val_loss = []
	for epoch in range(TCN1D_train_p['epochs']):

		model.train()
		optimizer.zero_grad()

		for x,y in train_loader:
			# (12, 1, 701)
			y_pred = model(x, y)
			loss_train = criterion(y_pred, y)
			loss_train.backward()
			torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)  # 梯度裁剪，防止梯度爆炸
			optimizer.step()
			train_loss.append(loss_train.item())

		for x, y in val_loader:
			# (3, 1, 701)
			model.eval()
			y_pred = model(x, y)
			loss_val = criterion(y_pred, y)
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

