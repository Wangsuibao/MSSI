import math
import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

from model.module import VGGExtractor, CNNExtractor, RNNLayer


class EncoderLAS(nn.Module):
	'''
		CNN获取局部邻域特征 ==>
		RNN获取序列特征==> output; hidden
		output --> Layer_nor -->dropout --> Linear --> tanh
		hidden --> cat --> Linear -->tanh
	'''
	def __init__(self, input_dim, feature_dim, enc_hid_dim, dec_hid_dim, dropout, prenet='cnn'):
		super().__init__()
		self.feature_dim = feature_dim
		self.vgg = prenet == 'vgg'
		self.cnn = prenet == 'cnn'
		'''
			1、input_size: 输入x的特征，2、hidden_size: 隐层的特征，3、num_layers:LSTM的堆叠数量， 4、bias: 是否使用b
			5、batch_fist: 输入输出的数据格式，True:（batch,seq,H_in）, 默认False(seq, batch, H_in)
			6、dropout：除最后一个层的每个LSTM层后增加一个dropout层，默认为0
			7、bidirectional: 是否为双向LSTM.默认False
		'''

		# 在LSTM之前可以增加一个卷积层作为特征获取
		if self.vgg:
			self.cnn_feature = VGGExtractor(input_size)
		if self.cnn:
			self.cnn_feature = CNNExtractor(input_dim, out_dim=self.feature_dim)

		# 需要确定
		self.rnn = RNNLayer(self.feature_dim, enc_hid_dim, bidirectional=True, dropout=0.7, layer_norm=True, proj=True)

		self.fc = nn.Linear(enc_hid_dim*2, dec_hid_dim)  # 双向GRU特征压缩, 用于decoder每个时间节点的输入

	def forward(self, src):
		'''
			所有的src:格式都是（Batch, channel, seq）

			seq:表示序列长度，input_dim:输入特征数，enc_hid_dim:编码器隐层特征数, bi：双向为2，num_layer:层数
			输入：input:（seq, batch, input_dim）; 
				  h_0:  (bi*num_layers, batch, enc_hid_dim)； 
			输出：output: (seq, batch, bi*enc_hid_dim);
				  h_n:  (bi*num_layers, batch, enc_hid_dim)-->(batch, 2*enc_hid_dim)--->fc-->(batch, dec_hid_dim)
		'''

		# 2、CNN局部特获取
		src = self.cnn_feature(src)
		src = src.permute(2,0,1).contiguous()  # （Batch, channel, seq）=>(seq, batch,channel)

		# 3、RNN序列特征获取，output,是否需要双向连接，Layer归一化，dropout, 全连接变换，tanh激活
		outputs, hidden = self.rnn(src)

		# 4、hidden维度变换
		# hidden[-2,:,:] 前向RNN隐层特征; hidden[-1,:,:]后向向RNN隐层特征
		# (2, Batch, enc_hid_dim)->(Batch, 2*enc_hid_dim) --fc-->(batch, dec_hid_dim)
		hidden = torch.tanh(self.fc(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)))
		
		# (seq, batch, bi*enc_hid_dim), (Batch, dec_hid_dim)
		return outputs, hidden


class AttentionLAS(nn.Module):
	'''
		需要输入的是
			hidden: encoder的hidden, 最后一个时间步的     : (batch, dec_hid_dim)
			encoder_outputs: encoder的outputs           : (seq, batch, bi*enc_hid_dim)
		输出：
			对encoder的outputs的每一个时间步的注意力系数   : (batch, seq)
	'''
	def __init__(self, enc_hid_dim, dec_hid_dim):
		super().__init__()
		self.attn = nn.Linear(2*enc_hid_dim+dec_hid_dim, dec_hid_dim)
		self.v = nn.Linear(dec_hid_dim, 1, bias=False)

	def forward(self, hidden, encoder_outputs):
		'''
			hidden : (batch, dec_hid_dim)， 处理(2*layers, Batch, enc_hid_dim)
					 在decoder循环的过程中，hidden由decoder更新，其维度都是(batch, dec_hid_dim)
					 重点： hidden随着decoder的每个时间步在更新
					 		to时，hidden来自encoder的输出，t1以后都来自decoder隐层。
			encoder_outputs (seq, batch, bi*enc_hid_dim)
		'''
		batch_size = encoder_outputs.shape[1]
		src_len = encoder_outputs.shape[0]

		# 重复hidden
		hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)  # (batch, seq, dec_hid_dim)
		encoder_outputs = encoder_outputs.permute(1, 0, 2)  # (batch, seq, bi*enc_hid_dim)

		# (batch, seq, bi*enc_hid_dim+dec_hid_dim) --> (batch, seq, dec_hid_dim)
		encoder_all_feature = self.attn(torch.cat((hidden, encoder_outputs), dim=2))
		energy = torch.tanh(encoder_all_feature)  
		attention = self.v(energy).squeeze(2)  # (batch, seq)

		return F.softmax(attention, dim=1)


class DecoderLAS(nn.Module):
	'''
		需要输入的是：
			1、encoder部分的最后一层循环输出的h_n 作为 decoder的h_0
			2、循环时，每次input是上次的output
	'''
	def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout, attention):
		super().__init__()
		
		self.output_dim = output_dim
		self.attention = attention

		self.rnn = nn.GRU(2*enc_hid_dim+output_dim, dec_hid_dim, num_layers=1, batch_first=False, bidirectional=False)

		self.fc_out = nn.Linear(2*enc_hid_dim+dec_hid_dim+output_dim, output_dim)  # 最后的拟合层：压缩不同来源feature部分
		
	def forward(self, input, hidden, encoder_outputs):
		'''
			hidden：  encoder部分的h_n或decoder中的h_i。    (bi*layers, batch, enc_hid_dim*2) --> (batch, dec_hid_dim)
			input：是一个时间节点的输出。 （1, batch, dec_hid_dim）
			encoder_outputs: 是整个时间序列的encoder的top输出，(seq, batch, enc_hid_dim*2)
		'''	

		'''
			第一步，每个时间步，使用隐层信息计算注意力系数，并计算t时刻需要的来之encoder的outputs的每一步记忆信息。
		'''
		a = self.attention(hidden, encoder_outputs)         # (batch, seq),经过softmax激活
		a = a.unsqueeze(1)                                  # (batch, 1, seq)
		encoder_outputs = encoder_outputs.permute(1, 0, 2)  # (batch, seq, enc_hid_dim*2)
		weighted = torch.bmm(a, encoder_outputs)            # (batch, 1, enc_hid_dim*2)  --- 最后两维矩阵乘法 ---
		weighted = weighted.permute(1, 0, 2)                # (1, batch, enc_hid_dim*2)  

		'''
			第二步：cat上一个时间步和注意力信息，作为input输入RNN
		'''
		rnn_input = torch.cat((input, weighted), dim=2)     # (1, batch, enc_hid_dim*2+output_dim)  叠加注意力w和上一个output
		# (seq=1, batch, dec_hid_dim), （n_layers=1, batch, dec_hid_dim）
		output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))  # 每次值输入t时刻

		'''
			第三步：cat， decoder的t时刻output, input, 记忆力信息。输入全连接层拟合输出
		'''
		all_feature = torch.cat((output, weighted, input),dim=2)  # (1, batch, dec_hid_dim+2*enc_hid_dim+output_dim)
		# 拟合decoder输出的特征（1, batch, channel）->（1, batch, 1）
		prediction = self.fc_out(all_feature)  # (1, batch, output_dim)

		'''
			第四步： prediction需要增加CNN，或者激活函数吗？
		'''

		# 需要注意的是，这里的三个输出都是某一时间节点的decoder输出，需要循环完成整个序列
		return prediction, hidden.squeeze(0)  # (batch, dec_hid_dim)


class seq2seqLAS(nn.Module):
	'''
		主要完成encoder的所有输出和decoder的所有输出的链接、合并方式
	'''
	def __init__(self, encoder, decoder, device):
		super().__init__()
		
		self.encoder = encoder
		self.decoder = decoder
		self.device = device
		# encoder的隐层enc_hid_dim和decoder的隐层dec_hid_dim必须一样，layers必须一样（只有一层所以hid_dim没有要求）
		
	def forward(self, src, trg, train=True, teacher_forcing_ratio=0.5):
		'''
			真实输入数据格式：
				src = （batch, H_in, seq）
				trg =  (batch, H_out, seq)
		'''
		trg = trg.permute(2,0,1).contiguous()  # （Batch, channel, seq）=>(seq, batch,channel)
		batch_size = trg.shape[1]
		trg_len = trg.shape[0]
		# 注意，decoder.output_dim == trg.shape[2], 如果是拟合，则为1，如果是分类，则为类别数。
		trg_vocab_size = self.decoder.output_dim  # 如果是岩性，这里不同岩性可以看成输出的词典
		
		# 储藏编码器的输出
		outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)  # (seq, batch, 1)

		# 1、编码器编码整个序列，获取最后的编码结果
		encoder_outputs, hidden = self.encoder(src)
		
		# 2、解码器的第一个输入需要能表示开始。直接输入0即可。（1, batch, trg_vocab_size）
		input = torch.zeros((1, batch_size, trg_vocab_size), dtype=trg.dtype, device=self.device)
		
		for t in range(1, trg_len):
			'''
				循环decoder,输出每个时间节点预测
				*** 完成t节点的运算，注意，在训练时，input来自label, 而在预测时，input来自上一个t节点的输出
			'''
			output, hidden = self.decoder(input, hidden, encoder_outputs)  # hidden: h_0是encoder的h_n, 后续是decoder的
			# decoder的预测（top输出）加入储存中
			outputs[t:t+1, :, :] = output
			
			# 是否使用 “teacher forcing”, 判断是训练状态，还是测试状态
			# 如果是“teacher forcing” 则label作为输入，如果不是，则上一个预测作为输入。
			teacher_force = random.random() < teacher_forcing_ratio
			if teacher_force and train:
				input = trg[t:t+1,:,:]
			else:
				input = output

		outputs = outputs.permute(1,2,0).contiguous()  # 转化成统一标准。（seq, batch, channel)-(batch, channel, seq）
		return outputs



if __name__ == '__main__':

	# 输入和输出的特征维度。当输入是地震多属性时，input_dim会增加，当输出是类别时，output_dim会增加
	INPUT_DIM = 1
	OUTPUT_DIM = 1

	# eocoder-decoder所有的隐层特征必须相等
	ENC_EMB_DIM = 256  # CNN扩增后的数据维度-encoder
	DEC_EMB_DIM = 256

	ENC_HID_DIM = 512  # RNN扩增后的数据维度-encoder
	DEC_HID_DIM = 512

	# 基本参数
	N_LAYERS = 1  # 这是固定的
	ENC_DROPOUT = 0.5
	DEC_DROPOUT = 0.5
	device = 'cpu'

		# 实例化模型
	attn = Attention(ENC_HID_DIM, DEC_HID_DIM)
	enc = EncoderLAS(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT)
	dec = DecoderLAS(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT, attn)

	model = seq2seqLAS(enc, dec, device).to(device)

	# 数据的组织维度是（Batch, Feature, seq）
	src = torch.normal(0.,1., size=(4, 1, 100))
	trg = torch.normal(0.,1., size=(4, 1, 100))
	output = model(src, trg)
	print('last ouput: ',output.shape)