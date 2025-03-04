
import torch
import torch.nn as nn
import random
import numpy as np

class CNNFeature(nn.Module):

	def __init__(self, input_dim, out_dim):
		super(CNNFeature, self).__init__()

		self.out_dim = out_dim


class RNNFeature(nn.Module):

	def __init__(self, input_dim, out_dim):

		super(RNNFeature, self).__init__()

		self.out_dim = out_dim


class FC_OUT(nn.Module):

	def __init__(self, input_dim, out_dim):
		super(FC_OUT, self).__init__()

		self.out_dim = out_dim


class CNN_GRU_Linear(nn.Module):
	def __init__(self, input_dim=1, hid_dim=256, output_dim=1, n_layers=2, dropout=0.75):
		super().__init__()
		self.hid_dim = hid_dim
		self.n_layers = n_layers
