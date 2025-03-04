
import torch
import torch.nn as nn
import random
import numpy as np


class Encoder(nn.Module):
	def __init__(self, input_dim, hid_dim, n_layers, dropout):
		super().__init__()
		self.hid_dim = hid_dim
		self.n_layers = n_layers
		self.feature_dim = 32

		'''
			1、input_size: 输入x的特征，2、hidden_size: 隐层的特征，3、num_layers:LSTM的堆叠数量， 4、bias: 是否使用b
			5、batch_fist: 输入输出的数据格式，True:（batch,seq,H_in）, 默认False(seq, batch, H_in)
			6、dropout：除最后一个层的每个LSTM层后增加一个dropout层，默认为0
			7、bidirectional: 是否为双向LSTM.默认False
		'''

		# 在LSTM之前可以增加一个卷积层作为特征获取
		self.cnn_feature = nn.Conv1d(input_dim, self.feature_dim, kernel_size=3, padding=1)

		self.rnn = nn.LSTM(self.feature_dim, hid_dim, n_layers, batch_first=False, dropout=dropout, bidirectional=False)


	def forward(self, src):
		'''
			所有的src:格式都是（Batch, channel, seq）

			seq:表示序列长度，H_in:输入特征数，H_cell:隐层特征数，H_out:输入特征数=隐层特征数, bi：双向为2，num_layer:层数
			输入：input:（seq, batch, H_in）; 
				  h_0:  (bi*num_layers, batch, H_out)； 
			      c_0:  (bi*num_layers, batch, H_cell);
			输出：output: (seq, batch, bi*H_out);
				  h_n:  (bi*num_layers, batch, H_out)； 输出信息
			      c_n:  (bi*num_layers, batch, H_cell);	记忆信息+更新信息 
		'''
		src = self.cnn_feature(src)
		src = src.permute(2,0,1).contiguous()  # （Batch, channel, seq）=>(seq, batch,channel)

		outputs, (hidden, cell) = self.rnn(src)
		
		# output:是叠置LSTM的最后一层输出。在encoder-decoder结构中不使用
		return outputs, hidden, cell

class Decoder(nn.Module):
	'''
		需要输入的是：
			1、encoder部分的最后一层循环输出的[c_n, h_n] 作为 decoder的c_0和h_0
			2、循环时，每次input是上次的output
	'''
	def __init__(self, output_dim, hid_dim, n_layers, dropout):
		super().__init__()
		
		self.output_dim = output_dim
		self.hid_dim = hid_dim
		self.n_layers = n_layers

		self.rnn = nn.LSTM(output_dim, hid_dim, n_layers, batch_first=False, dropout=dropout)

		self.fc_out = nn.Linear(hid_dim, output_dim)  # 最后的拟合层：压缩feature部分-----应该放在什么位置
		
	def forward(self, input, hidden, cell):
		'''
			c_0：  encoder部分的c_n。 (bi*num_layers, batch, H_out)
			h_0：  encoder部分的h_n。 (bi*num_layers, batch, H_out)
			input：是一个时间节点的输出。 （1, batch, Hy_in）
		'''	
		output, (hidden, cell) = self.rnn(input, (hidden, cell))
		'''
			c_i：  decoder的每一次输出。 (bi*num_layers, batch, H_out)
			h_i：  decoder的每一次输出。 (bi*num_layers, batch, H_out)
			output：是一个时间节点的输出。 （1, batch, Hy_out）
		'''	
		
		# 拟合decoder输出的特征（1, batch, channel）->（1, batch, 1），所以直接使用nn.Linear()
		prediction = self.fc_out(output)  # 其后是否需要激活函数，需要进一步测试？？
		
		# 需要注意的是，这里的三个输出都是某一时间节点的decoder输出，需要循环完成整个序列
		return prediction, hidden, cell


class seq2seq(nn.Module):
	'''
		主要完成encoder的所有输出和decoder的所有输出的链接、合并方式
	'''
	def __init__(self, encoder, decoder, device):
		super().__init__()
		
		self.encoder = encoder
		self.decoder = decoder
		self.device = device

		# encoder和decoder的隐层feature必须一样，layers必须一样
		assert encoder.hid_dim == decoder.hid_dim, \
			"Hidden dimensions of encoder and decoder must be equal!"
		assert encoder.n_layers == decoder.n_layers, \
			"Encoder and decoder must have equal number of layers!"
		
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
		E_output, hidden, cell = self.encoder(src)
		
		# 2、解码器的第一个输入需要能表示开始。直接输入0即可。（1, batch, H_out）
		input = torch.zeros((1, batch_size, trg_vocab_size), dtype=trg.dtype, device=self.device)
		
		for t in range(1, trg_len):
			'''
				循环decoder,输出每个时间节点预测
				输入包含三部分，h_i, c_i, output_i-1(上一个时间节点输出)
				*** 完成t节点的运算，注意，在训练时，input来自label, 而在预测时，input来自上一个t节点的输出
			'''

			output, hidden, cell = self.decoder(input, hidden, cell)
			# decoder的预测（top输出）加入储存中
			outputs[t:t+1, :, :] = output
			
			# 是否使用 “teacher forcing”, 判断是训练状态，还是测试状态.真预测阶段只能输入上一个时间点的输出，不能输入label
			# 如果是“teacher forcing” 则label作为输入，如果不是，则上一个预测作为输入。
			teacher_force = random.random() < teacher_forcing_ratio
			if teacher_force and train:
				# 训练时label作为输入
				input = trg[t:t+1,:,:]
			else:
				# 测试时上一个时间节点输出为下一个节点输入
				input = output

		outputs = outputs.permute(1,2,0).contiguous()  # 输出格式转化成（batch, channel, seq）
		return outputs



if __name__ == '__main__':

	# 输入和输出的特征维度。当输入是地震多属性时，input_dim会增加，当输出是类别时，output_dim会增加
	INPUT_DIM = 1
	OUTPUT_DIM = 1

	# eocoder-decoder所有的隐层特征必须相等
	HID_DIM = 512

	# 基本参数
	N_LAYERS = 2
	ENC_DROPOUT = 0.5
	DEC_DROPOUT = 0.5
	device = 'cpu'

	# 实例化模型
	enc = Encoder(INPUT_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
	dec = Decoder(OUTPUT_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)
	model = seq2seq(enc, dec, device).to(device)

	# 数据的组织维度是（Batch, Feature, seq）
	src = torch.normal(0.,1., size=(4, 1, 100))
	trg = torch.normal(0.,1., size=(4, 1, 100))
	output = model(src, trg)
	print('last ouput: ',output.shape)