from sklearn.mixture import GaussianMixture
from torch.nn import init
import numpy as np
import torch
import torch.utils.data as Data
import torch.nn as nn



# 一维正向卷积模块，用于构建FE的各层
def BNConv1dReLU(in_channels, out_channels, kernel_size, stride=1, padding=0):  #
    return nn.Sequential(
        nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                  padding=padding),  # 一维卷积
        nn.BatchNorm1d(out_channels),  # 批规范化
        nn.ReLU(inplace=True), )  # 激活函数


# ===================================================================
# 全连接模块，用于构建顶部特征提取器
def BNFCLReLU(in_size, out_size):
    return nn.Sequential(
        nn.Linear(in_size, out_size),
        nn.BatchNorm1d(out_size),   # 全连接层 fullconnected layers
        nn.ReLU(inplace=True), )


def BNFC(in_size, out_size):
    return nn.Sequential(
        nn.Linear(in_size, out_size),
        nn.BatchNorm1d(out_size),   # 全连接层 fullconnected layers
       )


# ===================================================================
class Flatten(nn.Module):
    def __init__(self):  # 构造函数，没有什么要做的
        super(Flatten, self).__init__()  # 调用父类构造函数

    def forward(self, input):  # 实现forward函数
        return input.view(input.size(0), -1)  # 保存batch维度，后面的维度全部压平，例如输入是28*28的特征图，压平后为784的向量





def FC_RELU(in_size, out_size):
    return nn.Sequential(
        nn.Linear(in_size, out_size),
        nn.ReLU(inplace=True), )





class Net_linear(nn.Module):
    def __init__(self):
        super(Net_linear, self).__init__()
        self.num_class = 9  # 输出层的神经元数
        self.num_in = 256  # 输入层的神经元数

        N_ClfSize = [self.num_in, int(0.25*self.num_in), self.num_class]
        self.Clf_block1 = FC_RELU(N_ClfSize[0], N_ClfSize[1])
        self.Clf_block2 = FC_RELU(N_ClfSize[1], N_ClfSize[2])

        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.normal_(m.weight, mean=0, std=0.02)
                init.constant_(m.bias, 0)


    def forward(self, x):
        x = x.to(torch.float32)
        x = self.Clf_block1(x)
        x = self.Clf_block2(x)

        output1 = x  # 分类结果输出

        return output1




class Net_trip(nn.Module):
	def __init__(self):
		super(Net_trip, self).__init__()
		N_FECh = [1, 32, 64, 128, 128, 128, 128]
		self.out_fea = 256 # 输出层的神经元数

		self.FE_block1 = nn.Sequential(
			BNConv1dReLU(in_channels=N_FECh[0], out_channels=N_FECh[1], kernel_size=64, stride=1, padding=0),  # 3072---3009
			nn.MaxPool1d(kernel_size=4, stride=4, padding=0))  # 752    卷积向下取整
		self.FE_block2 = nn.Sequential(
			BNConv1dReLU(in_channels=N_FECh[1], out_channels=N_FECh[2], kernel_size=3, stride=1, padding=0),  #750
			nn.Conv1d(in_channels=N_FECh[2], out_channels=N_FECh[2], kernel_size=3, stride=1, padding=0, dilation=2), #746
			nn.MaxPool1d(kernel_size=4, stride=4, padding=0))  # 186

		self.FE_block3 = nn.Sequential(
			BNConv1dReLU(in_channels=N_FECh[2], out_channels=N_FECh[3], kernel_size=3, stride=1, padding=0),  #184
			nn.Conv1d(in_channels=N_FECh[3], out_channels=N_FECh[3], kernel_size=3, stride=1, padding=0, dilation=2), #180
			nn.MaxPool1d(kernel_size=4, stride=4, padding=0))  # 45

		self.FE_block4 = nn.Sequential(
			BNConv1dReLU(in_channels=N_FECh[3], out_channels=N_FECh[4], kernel_size=3, stride=1, padding=0),  # 43
			nn.Conv1d(in_channels=N_FECh[4], out_channels=N_FECh[4], kernel_size=3, stride=1, padding=0, dilation=2),  # 39
			nn.MaxPool1d(kernel_size=4, stride=4, padding=1),   # 10
			Flatten())


		N_ClfSize = [1280,  self.out_fea]
		self.N_ClfSize = N_ClfSize
		self.Clf_block1 = BNFCLReLU(N_ClfSize[0], N_ClfSize[1])


	def forward_once(self, x):
		x = x.to(torch.float32)
		x = torch.reshape(x, (-1, 1, 3072))
		x = self.FE_block1(x)
		x = self.FE_block2(x)
		x = self.FE_block3(x)
		x = self.FE_block4(x)

		x = self.Clf_block1(x)

		output = x
		return output


	def forward(self, x1, x2, x3):
		output1 = self.forward_once(x1)
		output2 = self.forward_once(x2)
		output3 = self.forward_once(x3)

		return output1, output2, output3




class Net_trip_linear(nn.Module):
	def __init__(self):
		super(Net_trip_linear, self).__init__()
		N_FECh = [1, 32, 64, 128, 128, 128, 128]
		self.out_fea = 256 # 输出特征
		self.out_att = 9 # 输出属性


		N_ClfSize = [self.out_fea, 64, self.out_att]  # 1280
		self.N_ClfSize = N_ClfSize
		self.Clf_block1 = FC_RELU(N_ClfSize[0], N_ClfSize[1])
		self.Clf_block2 = FC_RELU(N_ClfSize[1], N_ClfSize[2])


		for m in self.modules():
			if isinstance(m, nn.Linear):
				init.normal_(m.weight, mean=0, std=0.02)
				init.constant_(m.bias, 0)

	def forward(self, x1, x2, x3):  # 原、正、负
		output1 = self.Clf_block1(x1)
		output2 = self.Clf_block1(x2)
		output3 = self.Clf_block1(x3)

		out = self.Clf_block2(output1)


		return output1, output2, output3, out




def pre_model_ori(test_pre_attribute, testlabel ,attribute_matrix_test):   # 自定义函数

	label_list = []

	for i in range(test_pre_attribute.shape[0]):
		pre_res = test_pre_attribute[i, :]
		dis = np.sum(np.square(attribute_matrix_test - pre_res), axis=1)    # 计算距离
		loc = dis.argmin()
		label_list.append(np.unique(testlabel)[loc])


	return test_pre_attribute,label_list





def match_l_c(cluster_center, sc, n):  # sc 得分，n 原型数
	# 以下为更改聚类标签语句

	sc = np.array(sc)
	sc = sc.reshape(n,n)

	# 找到最大的n个值的索引
	n_max = n*n
	# 将二维数组展平，然后找到展平后的数组中最大的n个值的索引
	flat_indices = np.argsort(sc, axis=None)[::-1][:n_max]
	# 将展平的索引转换为二维数组中的索引
	max_indices = np.column_stack(np.unravel_index(flat_indices, sc.shape))

	zero_vector = np.zeros(n)  # 初始化标签
	row =  max_indices[:, 0]
	col =  max_indices[:, 1]  # 标签

	vec = []
	zero_vector[row[0]] = col[0]
	vec.append(col[0])

	for i in range(len(row)-1):
		if not np.isin(col[i+1], vec):
			zero_vector[row[i+1]] = col[i+1]
			vec.append(col[i+1])

	vec_r = []
	vec_c = []
	zero_vector[row[0]] = col[0]
	vec_r.append(row[0])
	vec_c.append(col[0])

	for i in range(len(row) - 1):
		if not np.isin(row[i + 1], vec_r) and not np.isin(col[i + 1], vec_c):
			zero_vector[row[i + 1]] = col[i + 1]
			vec_r.append(row[i + 1])
			vec_c.append(col[i + 1])

	if len(zero_vector) != len(set(zero_vector)):
		print("Wrong, 数组中存在重复值")
		print(sc)
		print(vec_c)
		print(vec_r)
	# print("标签", zero_vector)

	if n==3:
		cluster_center_re = cluster_center[[zero_vector[0], zero_vector[1], zero_vector[2]], :]
	elif n==4:
		cluster_center_re = cluster_center[[zero_vector[0], zero_vector[1], zero_vector[2], zero_vector[3]], :]

	return cluster_center_re





def euclidean_dist(x, y):  # 计算矩阵的欧式距离
	"""
	Args:
	  x: pytorch Variable, with shape [m, d]
	  y: pytorch Variable, with shape [n, d]
	Returns:
	  dist: pytorch Variable, with shape [m, n]
	"""

	m, n = x.size(0), y.size(0)
	# xx经过pow()方法对每单个数据进行二次方操作后，在axis=1 方向（横向，就是第一列向最后一列的方向）加和，此时xx的shape为(m, 1)，经过expand()方法，扩展n-1次，此时xx的shape为(m, n)
	xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
	# yy会在最后进行转置的操作
	yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
	dist = xx + yy
	# torch.addmm(beta=1, input, alpha=1, mat1, mat2, out=None)，这行表示的意思是dist - 2 * x * yT
	dist.addmm_(x, y.t(), beta=1, alpha = -2)
	# clamp()函数可以限定dist内元素的最大最小范围，dist最后开方，得到样本之间的距离矩阵
	dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
	return dist



def cluster_clsGMM(fea, K_m, att_test):  #  K_N

	att_test = torch.tensor(att_test, dtype= torch.float)
	# kmeans = KMeans(n_clusters=K_m, random_state=9)

	model = GaussianMixture(n_components=K_m)
	yk_pred = model.fit_predict(fea)

	# cluster_centroids = torch.tensor(kmeans.cluster_centers_, dtype= torch.float)  # 质心
	cluster_centroids = torch.tensor(model.means_, dtype= torch.float)  # 质心

	sc = euclidean_dist(att_test,cluster_centroids)

	return yk_pred, sc,cluster_centroids



def gene_clu(gene_fea, num):

	times = int(len(gene_fea)/num)
	gene_fea = np.array(gene_fea.cpu().detach())

	gene_center = torch.empty((0, gene_fea.shape[1]))

	for t in np.arange(times):
		model = GaussianMixture(n_components=1)
		model.fit_predict(gene_fea[num * t : num * (t+1), :])
		cluster_cen1 = torch.tensor(model.means_, dtype=torch.float)  # 质心

		# model = KMeans(n_clusters=1)
		# model.fit_predict(gene_fea[num * t: num * (t + 1), :])
		# cluster_cen1= torch.tensor(model.cluster_centers_, dtype=torch.float)  # 质心

		gene_center = torch.cat((gene_center, cluster_cen1), dim=0)


	return gene_center




















