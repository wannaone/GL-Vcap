import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import mean_absolute_error
import numpy as np
import seaborn as sns
# sns.set(color_codes=True)
# sns.set()
import matplotlib.pyplot as plt
import  plotly.express as px
week_path = "D:/MasterStudents/2020/LvZhuanghu/data/runtimeData/05to09_49n_260week.csv"
week_np = np.loadtxt(week_path, dtype = np.float32, delimiter = ",", skiprows = 0)
y_max= week_np.T.max()
# np.delete(week_np,207,)
# data = np.load('D:/MasterStudents/2020/LvZhuanghu/non_val_ASTGCN/experiments/PEMS04/astgcn_r_h1d0w0_channel1_1.000000e-03/speed_output_epoch_43_test.npz')
T = 52
N= 49
# 一次实验的结果
# print(data.files)c  # ['input', 'prediction', 'data_target_tensor']cc
# data_target_tensor = data['data_target_tensor']
# prediction_data=data['prediction']  #23,9,4c
# data_target_tensor = data_target_tensor.reshape(T, N)
# pred_mean = prediction_data.reshape(T, N)
#pred_train0 = np.loadtxt('D:\MasterStudents/2020\LvZhuanghu\ex_result\ASTGCN/train/pred-train0.txt')
#pred_train1 = np.loadtxt('D:\MasterStudents/2020\LvZhuanghu\ex_result\ASTGCN/train/pred-train1.txt')
#pred_train2 = np.loadtxt('D:\MasterStudents/2020\LvZhuanghu\ex_result\ASTGCN/train/pred-train2.txt')
#val_pred = np.loadtxt('D:\MasterStudents/2020\LvZhuanghu\ex_result\ASTGCN/val/pred-val0.txt')
#pred_train_te = np.vstack((pred_train1,pred_train2))
#pred_train = np.vstack((pred_train0,pred_train_te))
# a_part = np.vstack((pred_train,val_pred))

# import true data and mean 多次试验的平均结果
pred_mean = np.loadtxt('D:\MasterStudents/2020\LvZhuanghu\ex_result\ASTGCN\predandtrue\pred/mean.txt')
pred_std = np.loadtxt('D:\MasterStudents/2020\LvZhuanghu\ex_result\ASTGCN\predandtrue\pred/std.txt')
#true_mean = np.loadtxt('D:\MasterStudents/2020\LvZhuanghu\ex_result\ASTGCN\predandtrue/true/mean.txt')
pred_mean = pred_mean.reshape(T, N)
# true_mean = true_mean.reshape(T, N)
pred_std = pred_std.reshape(T, N)
# b_part = np.vstack((val_pred,pred_mean))

lis  = np.arange(0,208).tolist()
inde = [0,1,2,6,7,8,9,10,45]
# for i in inde:
i = 45
plt.rcParams['font.sans-serif']=['Times New Roman'] # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False # 用来正常显示负号
plt.rcParams['savefig.dpi'] = 600
plt.figure(figsize=(16, 9), dpi=100)
# 设置横纵坐标的名称以及对应字体格式
font = {'family': 'Times New Roman',
'weight': 'normal',
'size': 32,
}
#plt.xlabel('Time', font)
plt.ylabel('Case', font)
xl = np.arange(156,260).tolist()
# 1:longchuan, 2:yijiang, 3:tenchong,7:lianghe,8:ruili,9:luxi,10:longling,:11:shidian,46:longyang,
# plt.plot(true_mean[:,0], 'ro-', label='ground truth')
plt.plot(week_np.T[lis,i], color = 'slategray', label='ground truth',linewidth=1.2)
plt.plot( np.arange(156,208),pred_mean[:,i],color = 'crimson' , label='prediction',linewidth=1.2)
# plt.plot(b_part[:,9], '.-', label='prediction')c
# plt errobarc
# plt.errorbar(x = np.arange(0,T),y = pred_mean[:,0],yerr=pred_std[:,0],ecolor="b")
plt.fill_between(np.arange(156,208),pred_mean[:,i]-pred_std[:,i],pred_mean[:,i]+pred_std[:,i],zorder=0,color = 'gray')
# plt.legend()  # 显示图例，即每条线对应 label 中的内容
# plt.vlines(208, 0, 100,colors='c',linestyles="dashed")
# plt.savefig("节点" + str(i + 1))
# plt.grid(True)  # 显示网格
my_xt = ['Jan 2005','Jan 2006','Jan 2007','Jan 2008','Jan 2009']
plt.xticks(np.arange(0,209,step = 52),my_xt,rotation = 0,fontproperties='Times New Roman', size=32)
plt.yticks(fontproperties='Times New Roman', size=32)
plt.legend(loc='upper right',fontsize=28)
# 1:longchuan, 2:yijiang, 3:tenchong,7:lianghe,8:ruili,9:luxi,10:longling,:11:shidian,46:longyang,
plt.savefig('longyang.eps', format='eps', dpi=1200)
plt.show()
