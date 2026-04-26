import random
import pandas as pd
import torch, time, os
import numpy as np
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from scipy import io
from scipy.io import savemat
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset
from torchvision.utils import save_image
import torch.nn.functional as F
from common.dataset import dataset_loader3_a_shuffle, data_trip4_improve, dataset_loader4_a_shuffle, dataset_loader4_e_shuffle
from common.network_and_loss import Net_trip
from Net_loss_diffusion1 import DDPM, Unet, EMA



class S_Generator(object):
	def __init__(self, d_n):
		'''
		初始化，定义超参数、数据集、网络结构等
		'''
		self.epoch = 60
		self.sample_num = 350  # 350  700 内存可能不够
		self.batch_size = 16  # 修改要
		self.lr = 0.0001
		self.n_T = 400      # 总共的step数
		self.device = 'cuda' if torch.cuda.is_available() else 'cpu'


		self.attribute_matrix_train = np.array(      # NEU
			[[0, 0, 0, 0, 0, 0, 0, 0, 0],  #
			 [0, 1, 0, 0, 1, 0, 0, 1, 0],
			 [1, 0, 0, 0, 0, 1, 1, 0, 0],
			 [0, 0, 1, 1, 0, 0, 0, 0, 1], ])


		self.attribute_matrix_test = np.array([[1, 1, 0, 0, 1, 1, 1, 0, 0],  # 内外  外滚  内滚
										  [0, 1, 1, 1, 1, 0, 0, 1, 0],
										  [1, 0, 1, 1, 0, 1, 1, 0, 0],
										  ])


		self.init_dataloader(d_n)
		self.sampler = DDPM(model=Unet(in_channels=1, n_feat = 32, n_label= 9), betas=(1e-4, 0.02), n_T=self.n_T, device=self.device).to(self.device)
		# self.sampler = DDPM(model=Unet(), betas=(1e-4, 0.02), n_T=self.n_T, device=self.device).to(self.device)
		self.optimizer = optim.Adam(self.sampler.model.parameters(), lr=self.lr)
		self.ema = EMA(model= self.sampler.model, decay= 0.995)
		self.ema.register()



	def init_dataloader(self, d_n):
		'''
		初始化数据集和dataloader
		'''

		num1 = 700
		num2 = 300
		num_out = 9


		# 数据加载 特征
		if d_n == 0:
			data1 = pd.read_csv('./Data/NEEPU/load_0/load_0_NC.csv_my.csv') # NEU
			data2 = pd.read_csv('./Data/NEEPU/load_0/load_0_OF.csv_my.csv')
			data3 = pd.read_csv('./Data/NEEPU/load_0/load_0_IF.csv_my.csv')
			data4 = pd.read_csv('./Data/NEEPU/load_0/load_0_BF.csv_my.csv')


		elif d_n == 1:
			data1 = pd.read_csv('./Data/NEEPU/load_1/load_1_NC.csv_my.csv')
			data2 = pd.read_csv('./Data/NEEPU/load_1/load_1_OF.csv_my.csv')
			data3 = pd.read_csv('./Data/NEEPU/load_1/load_1_IF.csv_my.csv')
			data4 = pd.read_csv('./Data/NEEPU/load_1/load_1_BF.csv_my.csv')


		elif d_n == 2:
			data1 = pd.read_csv('./Data/NEEPU/load_2/load_2_NC.csv_my.csv')
			data2 = pd.read_csv('./Data/NEEPU/load_2/load_2_OF.csv_my.csv')
			data3 = pd.read_csv('./Data/NEEPU/load_2/load_2_IF.csv_my.csv')
			data4 = pd.read_csv('./Data/NEEPU/load_2/load_2_BF.csv_my.csv')


		elif d_n == 3:
			data1 = pd.read_csv('./Data/NEEPU/load_3/load_3_NC.csv_my.csv')
			data2 = pd.read_csv('./Data/NEEPU/load_3/load_3_OF.csv_my.csv')
			data3 = pd.read_csv('./Data/NEEPU/load_3/load_3_IF.csv_my.csv')
			data4 = pd.read_csv('./Data/NEEPU/load_3/load_3_BF.csv_my.csv')


		sample_an1, label_1, label_a1 = dataset_loader4_a_shuffle(0, num1, data1, data2, data3, data4,    # 共有
		                                                                           self.attribute_matrix_train) # data1n, data2n, data3n, data4n,


		sample_an1 = sample_an1.to(self.device)


		train_dataset = TensorDataset(sample_an1 , label_1,  label_a1)

		self.train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True) # drop_last 删除不足的批次



	def train(self):

		self.sampler.train()
		print('训练开始!!')
		for ep in range(self.epoch):
			self.sampler.model.train()
			loss_mean = 0
			for i, (samples, labels, label_fu) in enumerate(self.train_dataloader):
				samples, labels, label_fu = samples.to(self.device), labels.to(self.device), label_fu.to(self.device)
				samples = torch.reshape(samples, (self.batch_size, 1, -1))  # 修改 batch


				loss = self.sampler(samples, label_fu)

				loss_mean += loss.item()
				self.optimizer.zero_grad()
				loss.backward()
				self.optimizer.step()
				self.ema.update()

			train_loss = loss_mean / len(self.train_dataloader)
			print('epoch:{}, loss:{:.4f}'.format(ep, train_loss))


			if (ep+1) % self.epoch == 0:
				self.generate_results_com(ep)
				self.generate_results_com2(ep)  # 内存不足分两次生成

				self.ema.restore()



	@torch.no_grad()
	def generate_results_com(self, epoch):
		self.ema.apply_shadow()
		self.sampler.eval()

		# 保存结果路径
		output_path = 'Data/Diffusion'
		if not os.path.exists(output_path):
			os.makedirs(output_path)

		tot_num_samples = self.sample_num
		image_frame_dim = int(np.floor(np.sqrt(tot_num_samples)))


		label_gene = torch.tensor(self.attribute_matrix_test, dtype=torch.float32)
		label_gene = label_gene.to(self.device)


		out = self.sampler.sample(tot_num_samples, (1, 3072), self.device, label_gene[0], 0.9).half() # 训练
		out_np = out.cpu().numpy().astype(np.float16)
		out_np = np.reshape(out_np, (self.sample_num, -1)).squeeze()
		del out


		out1 = self.sampler.sample(tot_num_samples, (1, 3072), self.device, label_gene[1], 0.9).half()
		out_np1 = out1.cpu().numpy().astype(np.float16)
		out_np1 = np.reshape(out_np1, (self.sample_num, -1)).squeeze()
		del out1


		out2 = self.sampler.sample(tot_num_samples, (1, 3072), self.device, label_gene[2], 0.9).half()
		out_np2 = out2.cpu().numpy().astype(np.float16)
		out_np2 = np.reshape(out_np2, (self.sample_num, -1)).squeeze()
		del out2


		# out3 = self.sampler.sample(tot_num_samples, (1, 3072), self.device, label_gene[3], 0.9).half()
		# out_np3 = out3.cpu().numpy().astype(np.float16)
		# out_np3 = np.reshape(out_np3, (self.sample_num, -1)).squeeze()
		# del out3


		# 保存为 .mat 文件
		save_path = os.path.join(output_path, "NEU_gene_%g_" %(epoch)  +  "_%g_" %(d_n)   +  ".mat") #+  "_%g_.mat" % (random.randrange(0, 999))
		savemat(save_path, {'OI': out_np, 'OB': out_np1, 'IB': out_np2}) # 复合故障



	@torch.no_grad()
	def generate_results_com2(self, epoch):
		self.ema.apply_shadow()
		self.sampler.eval()

		# 保存结果路径
		output_path = 'Data/Diffusion'
		if not os.path.exists(output_path):
			os.makedirs(output_path)

		tot_num_samples = self.sample_num
		image_frame_dim = int(np.floor(np.sqrt(tot_num_samples)))

		# label_traina =  torch.tensor( self.attribute_matrix_train, dtype=torch.float32 )  # 训练
		# label_traina = label_traina.to(self.device)
		label_gene = torch.tensor(self.attribute_matrix_test, dtype=torch.float32)
		label_gene = label_gene.to(self.device)



		out = self.sampler.sample(tot_num_samples, (1, 3072), self.device, label_gene[0], 0.9).half() # 训练
		out_np = out.cpu().numpy().astype(np.float16)
		out_np = np.reshape(out_np, (self.sample_num, -1)).squeeze()
		del out


		out1 = self.sampler.sample(tot_num_samples, (1, 3072), self.device, label_gene[1], 0.9).half()
		out_np1 = out1.cpu().numpy().astype(np.float16)
		out_np1 = np.reshape(out_np1, (self.sample_num, -1)).squeeze()
		del out1


		out2 = self.sampler.sample(tot_num_samples, (1, 3072), self.device, label_gene[2], 0.9).half()
		out_np2 = out2.cpu().numpy().astype(np.float16)
		out_np2 = np.reshape(out_np2, (self.sample_num, -1)).squeeze()
		del out2


		# out3 = self.sampler.sample(tot_num_samples, (1, 3072), self.device, label_gene[3], 0.9).half()
		# out_np3 = out3.cpu().numpy().astype(np.float16)
		# out_np3 = np.reshape(out_np3, (self.sample_num, -1)).squeeze()
		# del out3


		# 保存为 .mat 文件
		save_path = os.path.join(output_path, "NEU_gene2_%g_" %(epoch)  +  "_%g_" %(d_n)   +  ".mat") #+  "_%g_.mat" % (random.randrange(0, 999))
		savemat(save_path, {'OI': out_np, 'OB': out_np1, 'IB': out_np2}) # 复合故障 3





if __name__ == '__main__':
	data_num = [0, 1, 2, 3] # 0, 1, 2, 3

	for d_n in data_num:
		generator = S_Generator(d_n)
		generator.train()


