import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
def plt_2d(or_case,or_LST,or_SR,a,b,c,title,node):
    for i in range(node):
        plt.rcParams['font.sans-serif'] = ['Times New Roman']  # ??????????
        plt.rcParams['axes.unicode_minus'] = False  # ????????
        plt.rcParams['savefig.dpi'] = 600
        plt.figure(figsize=(16, 9), dpi=100)
        # ?????????????????
        font = {'family': 'Times New Roman',
                'weight': 'normal',
                'size': 32,
                }
        plt.xlabel('Time', font)
        plt.ylabel('Case', font)
        plt.title(title+"  node_" + str(i), font)
        plt.plot(or_case, color = 'blue', label=a,linewidth=1.2)
        plt.plot(or_LST, color = 'green', label=b,linewidth=1.2)
        plt.plot(or_SR, color = 'red', label=c,linewidth=1.2)
        plt.xticks( rotation=0, fontproperties='Times New Roman', size=32)
        plt.yticks(fontproperties='Times New Roman', size=32)
        plt.legend(loc='upper right', fontsize=28)
        plt.savefig("./result/"+title+"_.png",format="png",dpi=100)
        plt.show()

def plt_res(pre,target,title,node):
    for i in range(node):
        plt.rcParams['font.sans-serif'] = ['Times New Roman']  # ??????????
        plt.rcParams['axes.unicode_minus'] = False  # ????????
        plt.rcParams['savefig.dpi'] = 600
        plt.figure(figsize=(16, 9), dpi=100)
        # ?????????????????
        font = {'family': 'Times New Roman',
                'weight': 'normal',
                'size': 32,
                }
        plt.xlabel('Time', font)
        plt.ylabel('Case', font)
        plt.title(title+"  node_" + str(i), font)
        plt.plot(pre[:,i], color = 'blue', label='prediction',linewidth=1.2)
        plt.plot(target[:,i], color = 'green', label='true',linewidth=1.2)
        plt.xticks( rotation=0, fontproperties='Times New Roman', size=32)
        plt.yticks(fontproperties='Times New Roman', size=32)
        plt.legend(loc='upper right', fontsize=28)
        plt.savefig("./result/" + title + "_.png", format="png", dpi=100)
        plt.show()
def plt_res_mul(pre,target,node):
    for i in range(node):
        plt.rcParams['font.sans-serif'] = ['Times New Roman']  # ??????????
        plt.rcParams['axes.unicode_minus'] = False  # ????????
        plt.rcParams['savefig.dpi'] = 600
        plt.figure(figsize=(16, 9), dpi=100)
        # ?????????????????
        font = {'family': 'Times New Roman',
                'weight': 'normal',
                'size': 32,
                }
        plt.xlabel('Time', font)
        plt.ylabel('Case', font)
        plt.title("Node" + str(i), font)
        plt.plot(pre[:,i,0], color = 'blue', label='prediction',linewidth=1.2)
        plt.plot(target[:,i,0], color = 'green', label='true',linewidth=1.2)
        plt.xticks( rotation=2, fontproperties='Times New Roman', size=32)
        plt.yticks(fontproperties='Times New Roman', size=32)
        plt.legend(loc='upper right', fontsize=28)
        plt.show()
