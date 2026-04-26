import os
import random
import numpy as np
import pandas as pd
import scipy.stats
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats, io
from scipy.stats import gaussian_kde
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import MinMaxScaler


def target_distribution(x, mea, std):
	return stats.norm.pdf(x, mea, std)


# Metropolis-Hastings 算法
def metropolis_hastings_unknown(target_kde, start, iterations, proposal_width, burn_in, mea, std):
	samples = []
	current = start

	for _ in range(iterations):
		proposed = np.random.normal(current, proposal_width)
		# 使用核密度估计结果计算接受概率
		acceptance_ratio = target_kde(proposed, mea, std) / target_kde(current, mea, std)
		if np.random.rand() < acceptance_ratio:
			current = proposed

		samples.append(current)
	return np.array(samples[burn_in:])


# 假设已有数据
data_num = [0,1,2,3]

for d_n in data_num:

	# 数据加载 特征
	if d_n == 0:
		data1 = pd.read_csv('./Data/NEEPU/load_0/load_0_NC.csv_my.csv') # NEU
		data2 = pd.read_csv('./Data/NEEPU/load_0/load_0_OF.csv_my.csv')
		data3 = pd.read_csv('./Data/NEEPU/load_0/load_0_IF.csv_my.csv')
		data4 = pd.read_csv('./Data/NEEPU/load_0/load_0_BF.csv_my.csv')
		data5 = pd.read_csv('./Data/NEEPU/load_0/load_0_OI.csv_my.csv')
		data6 = pd.read_csv('./Data/NEEPU/load_0/load_0_BO.csv_my.csv')
		data7 = pd.read_csv('./Data/NEEPU/load_0/load_0_IB.csv_my.csv')

	elif d_n == 1:
		data1 = pd.read_csv('./Data/NEEPU/load_1/load_1_NC.csv_my.csv')
		data2 = pd.read_csv('./Data/NEEPU/load_1/load_1_OF.csv_my.csv')
		data3 = pd.read_csv('./Data/NEEPU/load_1/load_1_IF.csv_my.csv')
		data4 = pd.read_csv('./Data/NEEPU/load_1/load_1_BF.csv_my.csv')
		data5 = pd.read_csv('./Data/NEEPU/load_1/load_1_OI.csv_my.csv')
		data6 = pd.read_csv('./Data/NEEPU/load_1/load_1_BO.csv_my.csv')
		data7 = pd.read_csv('./Data/NEEPU/load_1/load_1_IB.csv_my.csv')

	elif d_n == 2:
		data1 = pd.read_csv('./Data/NEEPU/load_2/load_2_NC.csv_my.csv')
		data2 = pd.read_csv('./Data/NEEPU/load_2/load_2_OF.csv_my.csv')
		data3 = pd.read_csv('./Data/NEEPU/load_2/load_2_IF.csv_my.csv')
		data4 = pd.read_csv('./Data/NEEPU/load_2/load_2_BF.csv_my.csv')
		data5 = pd.read_csv('./Data/NEEPU/load_2/load_2_OI.csv_my.csv')
		data6 = pd.read_csv('./Data/NEEPU/load_2/load_2_BO.csv_my.csv')
		data7 = pd.read_csv('./Data/NEEPU/load_2/load_2_IB.csv_my.csv')

	elif d_n == 3:
		data1 = pd.read_csv('./Data/NEEPU/load_3/load_3_NC.csv_my.csv' )
		data2 = pd.read_csv('./Data/NEEPU/load_3/load_3_OF.csv_my.csv' )
		data3 = pd.read_csv('./Data/NEEPU/load_3/load_3_IF.csv_my.csv')
		data4 = pd.read_csv('./Data/NEEPU/load_3/load_3_BF.csv_my.csv' )
		data5 = pd.read_csv('./Data/NEEPU/load_3/load_3_OI.csv_my.csv')
		data6 = pd.read_csv('./Data/NEEPU/load_3/load_3_BO.csv_my.csv' )
		data7 = pd.read_csv('./Data/NEEPU/load_3/load_3_IB.csv_my.csv' )


	# 估计目标分布
	data1 = np.array(data1)
	data2 = np.array(data2)
	data3 = np.array(data3)
	data4 = np.array(data4)

	data1_mean = np.mean(data1, axis=1)[:, np.newaxis]
	data2_mean = np.mean(data2, axis=1)[:, np.newaxis]
	data3_mean = np.mean(data3, axis=1)[:, np.newaxis]
	data4_mean = np.mean(data4, axis=1)[:, np.newaxis]

	transfer = MinMaxScaler()  # 归一化
	data1_new = transfer.fit_transform(data1_mean)
	data2_new = transfer.fit_transform(data2_mean)
	data3_new = transfer.fit_transform(data3_mean)
	data4_new = transfer.fit_transform(data4_mean)

	# 拟合正态分布
	mea1, std1 = stats.norm.fit(data1_new)
	mea2, std2 = stats.norm.fit(data2_new)
	mea3, std3 = stats.norm.fit(data3_new)
	mea4, std4 = stats.norm.fit(data4_new)

	data1_var = np.var(data1, axis=1)[:, np.newaxis]
	data2_var = np.var(data2, axis=1)[:, np.newaxis]
	data3_var = np.var(data3, axis=1)[:, np.newaxis]
	data4_var = np.var(data4, axis=1)[:, np.newaxis]

	data1_newv = transfer.fit_transform(data1_var)
	data2_newv = transfer.fit_transform(data2_var)
	data3_newv = transfer.fit_transform(data3_var)
	data4_newv = transfer.fit_transform(data4_var)

	# 拟合正态分布
	mea1v, std1v = stats.norm.fit(data1_newv)
	mea2v, std2v = stats.norm.fit(data2_newv)
	mea3v, std3v = stats.norm.fit(data3_newv)
	mea4v, std4v = stats.norm.fit(data4_newv)


	# 参数设置
	start = 0  # 初始位置
	burn_in = 3000
	sam_num = 700
	iterations = burn_in + sam_num  # 迭代次数

	proposal_width = 1  # 提议分布的宽度

	# 生成样本
	x = np.linspace(-2, 2, 100)
	samples_NC1 = metropolis_hastings_unknown(target_distribution, start, iterations, proposal_width, burn_in, mea1, std1)
	samples_OF1 = metropolis_hastings_unknown(target_distribution, start, iterations, proposal_width, burn_in, mea2, std2)
	samples_IF1 = metropolis_hastings_unknown(target_distribution, start, iterations, proposal_width, burn_in, mea3, std3)
	samples_BF1 = metropolis_hastings_unknown(target_distribution, start, iterations, proposal_width, burn_in, mea4, std4)


	samples_NC2 = metropolis_hastings_unknown(target_distribution, start, iterations, proposal_width, burn_in, mea1v, std1v)
	samples_OF2 = metropolis_hastings_unknown(target_distribution, start, iterations, proposal_width, burn_in, mea2v, std2v)
	samples_IF2 = metropolis_hastings_unknown(target_distribution, start, iterations, proposal_width, burn_in, mea3v, std3v)
	samples_BF2 = metropolis_hastings_unknown(target_distribution, start, iterations, proposal_width, burn_in, mea4v, std4v)

	samples_NC = np.vstack((samples_NC1,samples_NC2))
	samples_OF = np.vstack((samples_OF1,samples_OF2))
	samples_IF = np.vstack((samples_IF1,samples_IF2))
	samples_BF = np.vstack((samples_BF1,samples_BF2))


	mdic = {'NC': samples_NC, 'OF': samples_OF, 'IF': samples_IF, 'BF': samples_BF}


	save_path = 'Data/MCMC'
	filepath = os.path.join(save_path, 'MCMC_NEU_load_' + "%g_" % (d_n) + "%g_" % (sam_num) + ".mat"  ) # +  "_%g_.mat" % (random.randrange(0, 999))

	io.savemat(filepath, mdic)
	print('Finish one')









