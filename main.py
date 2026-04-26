import os
import numpy as np
import pandas as pd
import torch
from scipy import io
from sklearn.metrics import accuracy_score
from torch import nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, TensorDataset
from common.dataset import dataset_loader3_a_shuffle, data_trip4_MCMC, data_gene_pro
from common.network_and_loss import Net_trip, Net_trip_linear, pre_model_ori, match_l_c, gene_clu, cluster_clsGMM
import warnings
warnings.filterwarnings("ignore")


def getloss(pred, x):
	loss = torch.pow(x - pred, 2).sum()
	loss /= x.size(0)
	return loss

label_testc = [12, 13, 23]

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
data_num = [1,2,3] # 0,1,2,3


Co_num = [0.1]
l_rate_num = [0.0001]  # 0.0001, 0.00005, 0.00007
cl_num = [0.15]  # 0.05, 0.1, 0.15, 0.2, 0.25


for d_n in data_num:
	for l_rate in l_rate_num:
		for C_co in Co_num:
			for cl_co in cl_num:
				for i in range(3):

					epoch = 1500
					BATCH_SIZE = 5000

					num1 = 700
					num2 = 300
					num_out = 9

					# 数据加载 特征
					if d_n == 0:
						data1 = pd.read_csv('./Data/NEEPU/load_0/load_0_NC.csv_my.csv')  # NEU
						data2 = pd.read_csv('./Data/NEEPU/load_0/load_0_OF.csv_my.csv')
						data3 = pd.read_csv('./Data/NEEPU/load_0/load_0_IF.csv_my.csv')
						data4 = pd.read_csv('./Data/NEEPU/load_0/load_0_BF.csv_my.csv')
						data5 = pd.read_csv('./Data/NEEPU/load_0/load_0_OI.csv_my.csv')
						data6 = pd.read_csv('./Data/NEEPU/load_0/load_0_BO.csv_my.csv')
						data7 = pd.read_csv('./Data/NEEPU/load_0/load_0_IB.csv_my.csv')

						data1c = io.loadmat(r'./Data\Diffusion\NEU_load_0_train_gene_59__.mat')
						data2c = io.loadmat(r'./Data\Diffusion\NEU_load_0_train_gene2_59__.mat')
						data_gene1 = io.loadmat(r'./Data\Diffusion\NEU_gene2_59__0_.mat')
						data_gene2 = io.loadmat(r'./Data\Diffusion\NEU_gene_59__0_.mat')


					elif d_n == 1:
						data1 = pd.read_csv('./Data/NEEPU/load_1/load_1_NC.csv_my.csv')
						data2 = pd.read_csv('./Data/NEEPU/load_1/load_1_OF.csv_my.csv')
						data3 = pd.read_csv('./Data/NEEPU/load_1/load_1_IF.csv_my.csv')
						data4 = pd.read_csv('./Data/NEEPU/load_1/load_1_BF.csv_my.csv')
						data5 = pd.read_csv('./Data/NEEPU/load_1/load_1_OI.csv_my.csv')
						data6 = pd.read_csv('./Data/NEEPU/load_1/load_1_BO.csv_my.csv')
						data7 = pd.read_csv('./Data/NEEPU/load_1/load_1_IB.csv_my.csv')

						data1c = io.loadmat(r'./Data\Diffusion\NEU_load_1_train_gene_59__.mat')
						data2c = io.loadmat(r'./Data\Diffusion\NEU_load_1_train_gene2_59__.mat')
						data_gene1 = io.loadmat(r'./Data\Diffusion\NEU_gene_59__1_.mat')
						data_gene2 = io.loadmat(r'./Data\Diffusion\NEU_gene2_59__1_.mat')


					elif d_n == 2:
						data1 = pd.read_csv('./Data/NEEPU/load_2/load_2_NC.csv_my.csv')
						data2 = pd.read_csv('./Data/NEEPU/load_2/load_2_OF.csv_my.csv')
						data3 = pd.read_csv('./Data/NEEPU/load_2/load_2_IF.csv_my.csv')
						data4 = pd.read_csv('./Data/NEEPU/load_2/load_2_BF.csv_my.csv')
						data5 = pd.read_csv('./Data/NEEPU/load_2/load_2_OI.csv_my.csv')
						data6 = pd.read_csv('./Data/NEEPU/load_2/load_2_BO.csv_my.csv')
						data7 = pd.read_csv('./Data/NEEPU/load_2/load_2_IB.csv_my.csv')

						data1c = io.loadmat(r'./Data\Diffusion\NEU_load_2_train_gene_59__.mat')
						data2c = io.loadmat(r'./Data\Diffusion\NEU_load_2_train_gene2_59__.mat')
						data_gene1 = io.loadmat(r'./Data\Diffusion\NEU_gene_59__2_.mat')
						data_gene2 = io.loadmat(r'./Data\Diffusion\NEU_gene2_59__2_.mat')


					elif d_n == 3:
						data1 = pd.read_csv('./Data/NEEPU/load_3/load_3_NC.csv_my.csv')
						data2 = pd.read_csv('./Data/NEEPU/load_3/load_3_OF.csv_my.csv')
						data3 = pd.read_csv('./Data/NEEPU/load_3/load_3_IF.csv_my.csv')
						data4 = pd.read_csv('./Data/NEEPU/load_3/load_3_BF.csv_my.csv')
						data5 = pd.read_csv('./Data/NEEPU/load_3/load_3_OI.csv_my.csv')
						data6 = pd.read_csv('./Data/NEEPU/load_3/load_3_BO.csv_my.csv')
						data7 = pd.read_csv('./Data/NEEPU/load_3/load_3_IB.csv_my.csv')

						data1c = io.loadmat(r'./Data\Diffusion\NEU_load_3_train_gene_59__.mat')
						data2c = io.loadmat(r'./Data\Diffusion\NEU_load_3_train_gene2_59__.mat')
						data_gene1 = io.loadmat(r'./Data\Diffusion\NEU_gene_59__3_.mat')
						data_gene2 = io.loadmat(r'./Data\Diffusion\NEU_gene2_59__3_.mat')

					attribute_matrix_train = np.array(
						[[0, 0, 0, 0, 0, 0, 0, 0, 0],
						 [0, 1, 0, 0, 1, 0, 0, 1, 0],
						 [1, 0, 0, 0, 0, 1, 1, 0, 0],
						 [0, 0, 1, 1, 0, 0, 0, 0, 1], ])

					attribute_matrix_test = np.array([[1, 1, 0, 0, 1, 1, 1, 0, 0],  # 内外  外滚  内滚
													  [0, 1, 1, 1, 1, 0, 0, 1, 0],
													  [1, 0, 1, 1, 0, 1, 1, 0, 0],
													  ])

					label_testc = [12, 13, 23]

					sample_an1, sample_na1, sample_po1, label_1, label_a1 = data_trip4_MCMC(0, num1, data1, data2,
																							data3, data4,  # 共有
																							data1c, data2c,
																							attribute_matrix_train)

					# sample_an1t, sample_na1t , label_po1t, label_1t, label_a1t = data_trip4(num2,(num2+100),(num2+200), data1, data2, data3, data4, attribute_matrix_train)
					sample_test, label_test, label_testa = dataset_loader3_a_shuffle(num2, data5, data6, data7,
																					 attribute_matrix_test)
					sample_gene = data_gene_pro(3, data_gene1, data_gene2)

					dataset_train = TensorDataset(sample_an1, sample_po1, sample_na1, label_1, label_a1)
					loader_train = DataLoader(dataset=dataset_train, batch_size=BATCH_SIZE)  # ,shuffle=True  不能shuffle

					sample_gene = sample_gene.to(device)
					sample_test = sample_test.to(device)
					sample_an1 = sample_an1.to(device)
					sample_po1 = sample_po1.to(device)
					sample_na1 = sample_na1.to(device)
					# sample_an1t =sample_an1t.to(device)

					path_m = 'NEU_model.pth'
					model = Net_trip().to(device)
					model.load_state_dict(torch.load(path_m))

					with torch.no_grad():
						out_an, out_po, out_na = model(sample_an1, sample_po1, sample_na1)
						# out_an = model(sample_an1)   #  , out_po , out_na
						# out_st, _ , _ = model(sample_an1t, sample_an1t, sample_an1t)

						out_c, _, _ = model(sample_test, sample_test, sample_test)
						out_gene, _, _ = model(sample_gene, sample_gene, sample_gene)

					# out_c = model(sample_test)

					model2 = Net_trip_linear().to(device)

					optimizer2 = torch.optim.Adam(model2.parameters(), lr=l_rate, weight_decay=1e-2)  # 可以换SGD
					# scheduler2 = StepLR(optimizer2, step_size=100, gamma=0.99)

					dataset_test = TensorDataset(out_c, label_test, label_testa)
					loader_test = DataLoader(dataset=dataset_test, batch_size=BATCH_SIZE)  # ,shuffle=True


					acc = []
					best_acc = 0
					loss_save = []
					# 分类训练
					for it in range(epoch):
						sample_fea = []
						sampletest_fea = []

						for step, (batch_an1, batch_na1, batch_po1, batch_y1, batch_y1a) in enumerate(loader_train):
							batch_an1, batch_na1, batch_po1, batch_y1, batch_y1a \
								= batch_an1.to(device), batch_na1.to(device), batch_po1.to(device), batch_y1.to(
								device), batch_y1a.to(device)

							output_an, output_po, output_na, att_out = model2(out_an, out_po, out_na)
							# att_out = model2(out_an)

							loss_clf = getloss(att_out, batch_y1a)  # w1, w2

							triplet_loss = nn.TripletMarginLoss(margin=5)
							loss_trip = triplet_loss(output_an, output_po, output_na)

							loss = loss_clf + C_co * loss_trip

							optimizer2.zero_grad()
							loss.backward()

							# torch.nn.utils.clip_grad_norm_(model2.parameters(), 1)  # 梯度裁剪收敛快
							optimizer2.step()

						# scheduler2.step()  # 衰减不能设置过低

						if it % 5 == 0:

							print('--------------------------------------------------------')
							print("epoch %d, loss_clf :%g" % (it, loss_clf))
							print("epoch %d, loss_trip :%g" % (it, loss_trip))

							loss_data = loss_clf.detach().cpu().numpy()
							loss_save.append(loss_data)

							# 测试
							# for step, (batch_x2, batch_y2, batch_y2a) in enumerate(loader_test):
							# model2.eval()

							_, _, _, att_test = model2(out_c, out_c, out_c)
							_, _, _, att_gene = model2(out_gene, out_gene, out_gene)

							att_test = att_test.cpu().detach()
							fea_test_att = np.array(att_test)

							sampletest_fea.extend(fea_test_att)
							sampletest_fea = np.array(sampletest_fea)


							y_pre_a, y_pre = pre_model_ori(test_pre_attribute=sampletest_fea, testlabel=label_testc,
														   attribute_matrix_test=attribute_matrix_test)

							yk_pred, sc, cluster_center = cluster_clsGMM(fea=sampletest_fea,
																		 K_m=len(attribute_matrix_test),
																		 att_test=attribute_matrix_test)  # k_m 原型数


							cluster_center_re = match_l_c(cluster_center, sc, len(attribute_matrix_test))  # 重构标签


							gene_center = gene_clu(att_gene, num1)

							cluster_center_np = cluster_center_re.numpy()
							gene_center_np = gene_center.numpy()

							gene_co = cl_co  # 修改

							re_proto = (1 - cl_co - gene_co) * attribute_matrix_test + cl_co * cluster_center_np + gene_co * gene_center_np

							y_pre_re_a, y_pre_re = pre_model_ori(test_pre_attribute=sampletest_fea,
																 testlabel=label_testc,
																 attribute_matrix_test=re_proto)

							y_pre_np = np.array(y_pre).squeeze()

							label_test_n = label_test.numpy()

							fusion_acc = accuracy_score(y_pre_re, label_test_n)


							acc.append(fusion_acc)

							if fusion_acc > best_acc:
								best_acc = fusion_acc
								best_epoch = it
								y_pre_best = y_pre_re

							print("epoch %d, best_acc :%g" % (it, best_acc))


					label_2 = label_test.numpy()
					best_acc_s = int(best_acc)
					mdic = {'y_pre': y_pre_best, 'y_true': label_2, 'acc': acc,
							'loss': loss_save}  # 'acc_o': acc_o,
					save_path = 'results'


					print(" acc :%g" % (best_acc))
					print('epoch %d' % (best_epoch))

					filepath = os.path.join(save_path, 'GMM_NEU_load_' + "%g_" % (d_n) + "%g_" % (l_rate) + "%g_" % (
						C_co) + "_clco_%g_" % cl_co
											+ "_ge_co_%g_" % gene_co + "__%g" % ((best_acc)) + " _ep_%g" % (
												best_epoch) + '.mat')  # + " _Km_%g" % (KM_n)

					io.savemat(filepath, mdic)






