
import torch
import torch.nn.functional as F

def cosine_distance_loss(x, y):
	cos_sim = F.cosine_similarity(x, y, dim=2)
	loss = 1 - cos_sim
	return loss.mean()

# 示例用法
x = torch.randn((10, 1, 128))  # 10个样本，每个样本128维
y = torch.randn((10, 1, 128))

loss = cosine_distance_loss(x, y)
print(loss)
