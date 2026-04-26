import random
import numpy as np
import pandas as pd
import scipy.io as scio
import torch



def dataset_loader4_a_shuffle(num1, num2,data1,data2,data3,data4,attribute_matrix):
	data1 = np.array(data1)
	data2 = np.array(data2)
	data3 = np.array(data3)
	data4 = np.array(data4)

	sample1 = data1[int(num1) : int(num2),0:3072]  # iloc
	label1 = data1[int(num1) : int(num2),3072]
	label_a1 = np.tile(attribute_matrix[0], ((num2-num1), 1))

	sample2 = data2[int(num1) : int(num2),0:3072]  # iloc
	label2 = data2[int(num1) : int(num2),3072]
	label_a2 = np.tile(attribute_matrix[1], ((num2-num1), 1))

	sample3 = data3[int(num1) : int(num2),0:3072]  # iloc
	label3 = data3[int(num1) : int(num2),3072]
	label_a3 = np.tile(attribute_matrix[2], ((num2-num1), 1))

	sample4 = data4[int(num1) : int(num2),0:3072]  # iloc
	label4 = data4[int(num1) : int(num2),3072]
	label_a4 = np.tile(attribute_matrix[3], ((num2-num1), 1))

	sample_1 = np.vstack((sample1, sample2, sample3, sample4))
	label_1 = np.hstack((label1, label2, label3, label4))
	label_a = np.vstack((label_a1, label_a2, label_a3, label_a4))

	index = [x for x in range(0, len(sample_1))]  # 数据打乱
	random.shuffle(index)
	sample_1 = sample_1[index]
	label_1 = label_1[index]
	label_a = label_a[index]

	sample = torch.as_tensor(sample_1, dtype=torch.float32)
	label = torch.as_tensor(label_1, dtype=torch.float32)
	label_a = torch.as_tensor(label_a, dtype=torch.float32) # , dtype=torch.long

	return sample, label ,label_a


def dataset_loader4_e_shuffle(num1, num2,data1,data2,data3,data4, dis_data,attribute_matrix):
	data1 = np.array(data1)
	data2 = np.array(data2)
	data3 = np.array(data3)
	data4 = np.array(data4)

	d_NC = dis_data['NC'].reshape(-1,2)
	d_OF = dis_data['OF'].reshape(-1,2)
	d_IF = dis_data['IF'].reshape(-1,2)
	d_BF = dis_data['BF'].reshape(-1,2)

	sample1 = data1[int(num1) : int(num2),0:3072]  # iloc
	label1 = data1[int(num1) : int(num2),3072]
	label_a1 = np.tile(attribute_matrix[0], ((num2-num1), 1))

	sample2 = data2[int(num1) : int(num2),0:3072]  # iloc
	label2 = data2[int(num1) : int(num2),3072]
	label_a2 = np.tile(attribute_matrix[1], ((num2-num1), 1))

	sample3 = data3[int(num1) : int(num2),0:3072]  # iloc
	label3 = data3[int(num1) : int(num2),3072]
	label_a3 = np.tile(attribute_matrix[2], ((num2-num1), 1))

	sample4 = data4[int(num1) : int(num2),0:3072]  # iloc
	label4 = data4[int(num1) : int(num2),3072]
	label_a4 = np.tile(attribute_matrix[3], ((num2-num1), 1))

	sample_1 = np.vstack((sample1, sample2, sample3, sample4))
	label_1 = np.hstack((label1, label2, label3, label4))
	label_a = np.vstack((label_a1, label_a2, label_a3, label_a4))
	label_dis = np.vstack((d_NC, d_OF, d_IF, d_BF))

	label_fu = np.hstack((label_a, label_dis))

	index = [x for x in range(0, len(sample_1))]  # 数据打乱
	random.shuffle(index)
	sample_1 = sample_1[index]
	label_1 = label_1[index]
	label_fu_sh = label_fu[index]

	sample = torch.as_tensor(sample_1, dtype=torch.float32)
	label = torch.as_tensor(label_1, dtype=torch.float32)
	# label_a = torch.as_tensor(label_a, dtype=torch.float16) # , dtype=torch.long
	label_fu = torch.as_tensor(label_fu, dtype=torch.float32)
	label_fu_sh = torch.as_tensor(label_fu_sh, dtype=torch.float32)


	return sample, label , label_fu, label_fu_sh




def dataset_loader3_a_shuffle(num,data1,data2,data3,attribute_matrix):

	data1 = np.array(data1)
	data2 = np.array(data2)
	data3 = np.array(data3)

	sample1 = data1[0:int(num),0:3072]  # iloc
	label1 = data1[0:int(num),3072]
	label_a1 = np.tile(attribute_matrix[0], (num, 1))    # 重复数组

	sample2 = data2[0:int(num),0:3072]  # iloc
	label2 = data2[0:int(num),3072]
	label_a2 = np.tile(attribute_matrix[1], (num, 1))

	sample3 = data3[0:int(num),0:3072]  # iloc
	label3 = data3[0:int(num),3072]
	label_a3 = np.tile(attribute_matrix[2], (num, 1))

	sample_1 = np.vstack((sample1, sample2, sample3))
	label_1 = np.hstack((label1, label2, label3))
	label_a = np.vstack((label_a1, label_a2, label_a3))

	index = [x for x in range(0, len(sample_1))]  # 数据打乱
	random.shuffle(index)
	sample_1 = sample_1[index]
	label_1 = label_1[index]
	label_a = label_a[index]

	sample = torch.as_tensor(sample_1, dtype=torch.float32)
	label = torch.as_tensor(label_1, dtype=torch.float32)
	label_a = torch.as_tensor(label_a, dtype=torch.float32)  # dtype=torch.long

	return sample, label ,label_a




def data_trip4_improve(num1, num2, data1, data2, data3, data4, data1n, data2n, data3n, data4n, attribute_matrix):
	data1 = np.array(data1)
	data2 = np.array(data2)
	data3 = np.array(data3)
	data4 = np.array(data4)

	data1n = np.array(data1n)
	data2n = np.array(data2n)
	data3n = np.array(data3n)
	data4n = np.array(data4n)

	sample1 = data1[int(num1): int(num2), 0:3072]  # iloc
	sample1p = data1n[int(num1): int(num2), 0:3072]  # iloc
	label1 = data1[int(num1): int(num2), 3072]
	label_a1 = np.tile(attribute_matrix[0], ((num2 - num1), 1))

	sample2 = data2[int(num1): int(num2), 0:3072]  # iloc
	sample2p = data2n[int(num1): int(num2), 0:3072]  # iloc
	label2 = data2[int(num1): int(num2), 3072]
	label_a2 = np.tile(attribute_matrix[1], ((num2 - num1), 1))

	sample3 = data3[int(num1): int(num2), 0:3072]  # iloc
	sample3p = data3n[int(num1): int(num2), 0:3072]  # iloc
	label3 = data3[int(num1): int(num2), 3072]
	label_a3 = np.tile(attribute_matrix[2], ((num2 - num1), 1))

	sample4 = data4[int(num1): int(num2), 0:3072]  # iloc
	sample4p = data4n[int(num1): int(num2), 0:3072]  # iloc
	label4 = data4[int(num1): int(num2), 3072]
	label_a4 = np.tile(attribute_matrix[3], ((num2 - num1), 1))

	sample_anchor = np.vstack((sample1, sample2, sample3, sample4))
	sample_posi = np.vstack((sample1p,sample2p,sample3p,sample4p))

	label_1 = np.hstack((label1, label2, label3, label4))
	label_a = np.vstack((label_a1, label_a2, label_a3, label_a4))

	sample_nage = np.vstack((sample4,sample3,sample2,sample1))

	index = [x for x in range(0, len(sample_anchor))]  # 数据打乱
	random.shuffle(index)
	sample_anchor = sample_anchor[index]
	sample_nage = sample_nage[index]
	sample_posi = sample_posi[index]

	label_1 = label_1[index]
	label_a = label_a[index]

	sample_an = torch.as_tensor(sample_anchor)
	sample_na = torch.as_tensor(sample_nage)
	sample_po = torch.as_tensor(sample_posi)
	label_a = torch.as_tensor(label_a)
	label = torch.as_tensor(label_1, dtype=torch.long)

	return sample_an, sample_na, sample_po, label, label_a


def data_trip4_MCMC(num1, num2, data1, data2, data3, data4, data1c, data2c, attribute_matrix):
	data1 = np.array(data1)
	data2 = np.array(data2)
	data3 = np.array(data3)
	data4 = np.array(data4)


	sample1 = data1[int(num1): int(num2), 0:3072]  # iloc
	label1 = data1[int(num1): int(num2), 3072]
	label_a1 = np.tile(attribute_matrix[0], ((num2 - num1), 1))

	sample2 = data2[int(num1): int(num2), 0:3072]  # iloc
	label2 = data2[int(num1): int(num2), 3072]
	label_a2 = np.tile(attribute_matrix[1], ((num2 - num1), 1))

	sample3 = data3[int(num1): int(num2), 0:3072]  # iloc
	label3 = data3[int(num1): int(num2), 3072]
	label_a3 = np.tile(attribute_matrix[2], ((num2 - num1), 1))

	sample4 = data4[int(num1): int(num2), 0:3072]  # iloc
	label4 = data4[int(num1): int(num2), 3072]
	label_a4 = np.tile(attribute_matrix[3], ((num2 - num1), 1))


	sample_anchor = np.vstack((sample1, sample2, sample3, sample4))


	sample1p = np.vstack((data1c['NC'], data2c['NC']))
	sample2p = np.vstack((data1c['OF'], data2c['OF']))
	sample3p = np.vstack((data1c['IF'], data2c['IF']))
	sample4p = np.vstack((data1c['BF'], data2c['BF']))


	sample_posi = np.vstack((sample1p,sample2p,sample3p,sample4p))

	label_1 = np.hstack((label1, label2, label3, label4))
	label_a = np.vstack((label_a1, label_a2, label_a3, label_a4))

	sample_nage = np.vstack((sample4,sample3,sample2,sample1))

	index = [x for x in range(0, len(sample_anchor))]  # 数据打乱
	random.shuffle(index)
	sample_anchor = sample_anchor[index]
	sample_nage = sample_nage[index]
	sample_posi = sample_posi[index]

	label_1 = label_1[index]
	label_a = label_a[index]

	sample_an = torch.as_tensor(sample_anchor)
	sample_na = torch.as_tensor(sample_nage)
	sample_po = torch.as_tensor(sample_posi)
	label_a = torch.as_tensor(label_a)
	label = torch.as_tensor(label_1, dtype=torch.long)

	return sample_an, sample_na, sample_po, label, label_a




def data_gene_pro(num, data_gene1, data_gene2):


	sample1 = np.vstack((data_gene1['OI'], data_gene2['OI']))
	sample2 = np.vstack((data_gene1['OB'], data_gene2['OB']))
	sample3 = np.vstack((data_gene1['IB'], data_gene2['IB']))

	sample1 = torch.as_tensor(sample1)
	sample2 = torch.as_tensor(sample2)
	sample3 = torch.as_tensor(sample3)


	if num == 4:
		sample4 = np.vstack((data_gene1['BIO'], data_gene2['BIO']))
		sample4 = torch.as_tensor(sample4)
		sample_gene = torch.stack([sample1, sample2, sample3, sample4], dim = 0)
		return sample_gene

	else:
		return torch.stack([sample1, sample2, sample3],dim = 0)
































