
import torch
import torch.nn as nn

class FCNN(nn.Module):
	'''
		需要注意的模块设置
			1、最后一层的激活函数。
				input: (*, Hin) , Hin = in_features
				output: (*, Hout), Hout = out_features
				weight: 学习权重：(out_features, in_features)

	'''
	def __init__(self):
		super(FCNN, self).__init__()
		self.hidden1=nn.Sequential(
				nn.Linear(in_features=1, out_features=30, bias=True),
				nn.ReLU())
		self.hidden2=nn.Sequential(
				nn.Linear(in_features=30, out_features=10, bias=True),
				nn.ReLU())
		self.hidden3=nn.Sequential(
				nn.Linear(in_features=10, out_features=1, bias=True),
				nn.Sigmoid())

	def forward(self,x):
		fc1=self.hidden1(x)
		fc2=self.hidden2(fc1)
		output=self.hidden3(fc2)
		return output


if __name__ == '__main__':
	FCNN = FCNN()
	x = torch.randn(size=(10, 1))
	y = FCNN(x)
	print(y)