# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from statsmodels.tsa.seasonal import seasonal_decompose
from model.VCAP_L import VcapModel
import numpy as np
# from lib.utils import scaled_Laplacian, cheb_polynomial
import datetime

from model.Mylstm import LSTMTest


class Linear_self(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        b, n, f, t = x.shape
        outputs_batch = []
        for batch_step in range(b):
            x_batch = x[batch_step, :, :, :]
            outputs_seq = []
            for i in range(t):
                x_seq = x_batch[:, :, i]  # n,f
                out_seq = self.linear(x_seq)
                outputs_seq.append(out_seq)
            outputs_seq_ten = torch.stack(outputs_seq, 2)  # n f_out t
            outputs_batch.append(outputs_seq_ten)
        outputs_batch_ten = torch.stack(outputs_batch, 0)  # b n f_out t
        return outputs_batch_ten


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """

    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)
        # b,n,f,t

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class DLinear(nn.Module):
    """
    DLinear
    """

    def __init__(self, seq_len, pred_len, individual, enc_in):
        super(DLinear, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len

        # Decompsition Kernel Size
        kernel_size = 1
        self.decompsition = series_decomp(kernel_size)
        self.individual = individual
        self.channels = enc_in

        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()
            self.Linear_Decoder = nn.ModuleList()
            for i in range(self.channels):
                self.Linear_Seasonal.append(nn.Linear(self.seq_len, self.pred_len))
                self.Linear_Seasonal[i].weight = nn.Parameter(
                    (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
                self.Linear_Trend.append(nn.Linear(self.seq_len, self.pred_len))
                self.Linear_Trend[i].weight = nn.Parameter(
                    (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
                self.Linear_Decoder.append(nn.Linear(self.seq_len, self.pred_len))
        else:
            self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
            self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)
            self.Linear_Decoder = nn.Linear(self.seq_len, self.pred_len)
            self.Linear_Seasonal.weight = nn.Parameter((1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
            self.Linear_Trend.weight = nn.Parameter((1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))

    def forward(self, x):
        # x: [Batach, Input length, Channal]
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init, trend_init = seasonal_init.permute(0, 2, 1), trend_init.permute(0, 2, 1)
        if self.individual:
            seasonal_output = torch.zeros([seasonal_init.size(0), seasonal_init.size(1), self.pred_len],
                                          dtype=seasonal_init.dtype).to(seasonal_init.device)
            trend_output = torch.zeros([trend_init.size(0), trend_init.size(1), self.pred_len],
                                       dtype=trend_init.dtype).to(trend_init.device)
            for i in range(self.channels):
                seasonal_output[:, i, :] = self.Linear_Seasonal[i](seasonal_init[:, i, :])
                trend_output[:, i, :] = self.Linear_Trend[i](trend_init[:, i, :])
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)

        x = seasonal_output + trend_output
        return x.permute(0, 2, 1)  # to [Batach, Output length, Channal]


class LSTMTest(nn.Module):
    def __init__(self, DEVICE, input_size, num_layers, output_size):
        super().__init__()
        self.input_size = input_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.num_directions = 1
        self.lstmq = nn.LSTM(self.input_size, self.output_size, self.num_layers, batch_first=True).to(DEVICE)
        self.DEVICE = DEVICE

        # self.linear = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, DEVICE, input_seq):
        # input_seq = input_seq.permute(3,1,2,0)
        # input_seq :(B,N,F,T)
        (batch_size, num_of_vertices, num_of_features, num_of_timesteps) = input_seq.shape
        target = []
        for time_step in range(batch_size):
            input_ = input_seq[time_step, :, :, :]  # (B,N,F,T)-->(N,F,T)
            input_ = input_.permute(0, 2, 1)
            h_0 = torch.randn(self.num_directions * self.num_layers, num_of_vertices, self.output_size).to(DEVICE)
            c_0 = torch.randn(self.num_directions * self.num_layers, num_of_vertices, self.output_size).to(DEVICE)
            output, _ = self.lstmq(torch.tensor(input_, dtype=torch.float), (h_0, c_0))  # (N,T,F)
            output = output.permute(0, 2, 1)  # (N,F_out,T)
            target.append(output)
        return torch.stack(target, 0)  # (B,N,F,T)


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, DEVICE, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)).to(DEVICE))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha).to(DEVICE)

    def forward(self, x, adj):
        '''
        Parameters
        ----------
        input: mx.ndarray, graph signal matrix
           shape is (batch_size, N, F, T_{r-1}), F is the num of features
        adj:N*N
        Returns
        -------
        '''

        (batch_size, num_of_vertices, num_of_features, num_of_timesteps) = x.shape  # (B,N,F,T)
        outputs_Time = []
        for time_step in range(num_of_timesteps):
            graph_signal = x[:, :, :, time_step]  # shape is (batch_size, N, F)
            outputs_batch = []
            for i in range(batch_size):
                h = graph_signal[i, :, :]  # shape is (batch_size, N, F)
                Wh = torch.mm(h, self.W)  # h.shape: (N, in_features), Wh.shape: (N, out_features)
                e = self._prepare_attentional_mechanism_input(Wh)  # n*N
                zero_vec = -9e15 * torch.ones_like(e)  # n*n
                attention = torch.where(adj > 0, e, zero_vec)
                attention = F.softmax(attention, dim=1)
                attention = F.dropout(attention, self.dropout, training=self.training)
                h_prime = torch.matmul(attention, Wh)
                outputs_batch.append(h_prime)  # list有4个(N,F)
            # outputs_batch->(B,N,F)
            outputs_batch_ten = torch.stack(outputs_batch, 0)  # （b,n,f）
            outputs_Time.append(outputs_batch_ten)  # list有4个(b,N,F)
        outputs_time_ten = torch.stack(outputs_Time, 0)  # (T,B,N,F_out)
        outputs = outputs_time_ten.permute(1, 2, 3, 0)  # (B,N,F_out,T)
        # outputs_batch_ten = outputs_batch_ten.permute(0,3,2,1).permute(0,2,1,3)
        # outputs = outputs_batch_ten
        if self.concat:
            return F.elu(outputs)
        else:
            return outputs

    def _prepare_attentional_mechanism_input(self, Wh):
        # Wh.shape (B, N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        # broadcast add
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GAT(nn.Module):
    def __init__(self, DEVICE, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(DEVICE, nfeat, nhid, dropout, alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        self.out_att = GraphAttentionLayer(DEVICE, nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        batch, n, f, t = x.shape
        x = F.dropout(x, self.dropout, training=self.training)  # b,n,f,t
        x1 = torch.stack([att(x, adj) for att in self.attentions], dim=2)  # 应该是B N Nhid*Nheads T
        x1 = x1.reshape(batch, n, -1, t)
        x2 = F.dropout(x1, self.dropout, training=self.training)
        x3 = self.out_att(x2, adj)
        # x3 = F.elu(x3)
        return x3


class ASTGCN_block(nn.Module):
    def __init__(self, DEVICE, in_channels, hid_feature, out_feature, nheads, lstm_out, dropout, alpha, adj_mx):
        super(ASTGCN_block, self).__init__()
        self.adj = torch.from_numpy(adj_mx).to(DEVICE)
        self.GAt = GAT(DEVICE, in_channels, hid_feature, out_feature, dropout, alpha, nheads)
        self.LSTM = LSTMTest(DEVICE, out_feature, 1, lstm_out).to(DEVICE)
        self.residual_conv = nn.Conv2d(in_channels, lstm_out, kernel_size=(1, 1), stride=(1, 1)).to(DEVICE)
        self.ln = nn.LayerNorm(lstm_out).to(DEVICE)  # 需要将channel放到最后一个维度上
        self.DEVICE = DEVICE
        self.to(DEVICE)

    def forward(self, DEVICE, x):
        '''
        :param x: (batch_size, N, F_in, T)
        :return: (batch_size, N, nb_time_filter, T)
        '''
        batch_size, num_of_vertices, num_of_features, num_of_timesteps = x.shape
        # 用GAT
        spatial_gcn = self.GAt(x, self.adj)  # return(B,N,out_feature,T)
        lstm_temp = self.LSTM(DEVICE, spatial_gcn).permute(0, 2, 1, 3)  # return(B,N,lstm_out,T)->(b,lstm_out,N,T)
        # (b,N,in_channels,T)->(b,in_channels,N,T) 用(1,1)的卷积核去做->(b,lstm_out,N,T)
        x_residual = self.residual_conv(x.permute(0, 2, 1, 3))
        x_residual = self.ln(F.relu((x_residual + lstm_temp)).permute(0, 3, 2, 1)).permute(0, 2, 3, 1)
        # (b,F,N,T)->(b,T,N,F) -ln-> (b,T,N,F)->(b,N,F,T)=(b,N,lstm_out,T)
        return x_residual


class ASTGCN_submodule(nn.Module):
    def __init__(self, DEVICE, nb_block, in_channels, hid_feature, out_feature, nheads, lstm_out, dropout, alpha,
                 adj_mx, time_strides, num_for_predict):
        super(ASTGCN_submodule, self).__init__()
        self.BlockList = nn.ModuleList(
            [ASTGCN_block(DEVICE, in_channels, hid_feature, out_feature, nheads, lstm_out, dropout, alpha, adj_mx)]).to(
            DEVICE)
        self.BlockList.extend(
            [ASTGCN_block(DEVICE, lstm_out, 8, 32, nheads, 1, dropout, alpha, adj_mx) for _ in range(nb_block - 1)]).to(
            DEVICE)
        # (b,N,F_out,T)
        self.final_conv = nn.Conv2d(int(time_strides), num_for_predict, kernel_size=(1, 1)).to(DEVICE)
        # 第一种

        # 第二种
        self.x2_linear = Linear_self(3, 4).to(DEVICE)
        self.final_linear_1 = Linear_self(8, 10).to(DEVICE)
        self.final_linear_2 = Linear_self(10, 1).to(DEVICE)
        # 第三种种
        self.ser_dep = series_decomp(9)
        self.Infin_linear = Linear_self(2, 10).to(DEVICE)
        self.Outfin_linear = Linear_self(10, 1).to(DEVICE)
        self.Feat_lineat = Linear_self(4,1).to(DEVICE)

        # self.Outfin_linear = Linear_self(time_strides, num_for_predict).to(DEVICE)
        self.time_strides = time_strides
        self.vcapLay = VcapModel(1, 1)
        self.DEVICE = DEVICE
        self.to(DEVICE)

    def forward(self, DEVICE, x):
        '''
        :param x: (B, N_nodes, F_in, T_in)
        :return: (B, N_nodes, F_out(1),T_out)
        '''
    # 有时间卷积
      # 第一种 vcap
        '''
        x1 = x[:, :, 0, :].unsqueeze(2)
        x2 = x[:, :, 1:3, :]
        for block in self.BlockList:
            x1 = block(DEVICE, x1)  # b n f(1) T(4)
        x2 = self.vcapLay(x2) # b n f(1) T(4)
        # 将vcap和case特征拼接、经过两个线性层变换
        x_concat = torch.cat([x1, x2], dim=2)  # b n f(2) T(4)
        x2_linear = self.Infin_linear(x_concat)  # b n f(50) T(4)
        x2_linear = F.relu(x2_linear)
        x2_linear = self.Outfin_linear(x2_linear)  # b n f(1) T
        # b n f(1) T(1) ---> b,N,T(1)
        output = self.final_conv(x2_linear.permute(0,3,1,2)).permute(0,2,3,1)[:,:,:,-1]
        return output
        '''

        # 第三种 vcap
        '''b, n, f, t = x.shape
        x1 = x[:, :, 0, :].unsqueeze(2)
        x2 = x[:, :, 1:3, :]
        for block in self.BlockList:
            x1 = block(DEVICE, x1)  # b n f(1) T(1)
        x1 = self.final_conv(x1.permute(0,3,1,2)).permute(0,2,3,1)
        # x1 = x1[:, :, :,-1].unsqueeze(3)
        x2 = self.vcapLay(x2)[:, :, :, -1].unsqueeze(3)  # b,n,f(1),T(4) --> b n f(1) T(1)
        # 将vcap和case特征拼接、经过两个线性层变换
        x_concat = torch.cat([x1, x2], dim=2)  # b n f(2) t
        x2_linear = self.Infin_linear(x_concat)  # b n f(50) t
        x2_linear = F.relu(x2_linear)
        x2_linear = self.Outfin_linear(x2_linear)  # b n f(1) t
        # b n f(1) t ---> b,N,T(1)
        output = x2_linear[:, :, :, -1]
        # (b,N,F,T)->(b,T,N,F)-conv<1,F>->(b,c_out*T,N,1)->(b,c_out*T,N)->(b,N,T)
        return output
        '''
        
        # 第五种 vcap 将F变成4再变成1，vcap变成1，cat后再LL再改变T：效果不错
        x1 = x[:, :, 0, :].unsqueeze(2)
        x2 = x[:, :, 1:3, :]
        for block in self.BlockList:
            x1 = block(DEVICE, x1)  # b n f(4) T(4)
        x1 = self.Feat_lineat(x1)
        x2 = self.vcapLay(x2)  # b n f(1) T(4)
        # 将vcap和case特征拼接、经过两个线性层变换
        x_concat = torch.cat([x1, x2], dim=2)  # b n f(2) T(4)
        x2_linear = self.Infin_linear(x_concat)  # b n f(50) T(4)
        # x2_linear = F.relu(x2_linear)
        x2_linear = self.Outfin_linear(x2_linear)  # b n f(1) T(4)
        # b n f(1) T(1) ---> b,N,T(1)
        output = self.final_conv(x2_linear.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)[:, :, :, -1]
        return output
        

        # 第六种 将F变成4再变成1，vcap变成1,T变成1，cat后再LL：效果一般
        '''
        x1 = x[:, :, 0, :].unsqueeze(2)
        x2 = x[:, :, 1:3, :]
        for block in self.BlockList:
            x1 = block(DEVICE, x1)  # b n f(4) T(4)
        x1 = self.Feat_lineat(x1)  # b n f(1) T(4)
        x1 = x1[:,:,:,-1].unsqueeze(3) # b n f(1) T(1)
        x2 = self.vcapLay(x2)[:,:,:,-1].unsqueeze(3)  # b n f(1) T(1)
        # 将vcap和case特征拼接、经过两个线性层变换
        x_concat = torch.cat([x1, x2], dim=2)  # b n f(2) T(1)
        x2_linear = self.Infin_linear(x_concat)  # b n f(50) T(1)
        x2_linear = F.relu(x2_linear)
        x2_linear = self.Outfin_linear(x2_linear)  # b n f(1) T
        # b n f(1) T(1) ---> b,N,T(1)
        output = x2_linear[:,:,:,-1]
        # output = self.final_conv(x2_linear.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)[:, :, :, -1]
        return output
        '''

# 含有时间卷积的模型
class ASTGCN_TimeCon(nn.Module):
    def __init__(self, DEVICE, nb_block, in_channels, hid_feature, out_feature, nheads, lstm_out, dropout, alpha,
                 adj_mx, time_strides, num_for_predict):
        super(ASTGCN_TimeCon, self).__init__()
        self.BlockList = nn.ModuleList(
            [ASTGCN_block(DEVICE, in_channels, hid_feature, out_feature, nheads, lstm_out, dropout, alpha, adj_mx)]).to(
            DEVICE)
        self.BlockList.extend(
            [ASTGCN_block(DEVICE, lstm_out, 8, 32, nheads, 1, dropout, alpha, adj_mx) for _ in range(nb_block - 1)]).to(
            DEVICE)
        # (b,N,F_out,T)
        self.final_conv = nn.Conv2d(int(time_strides), num_for_predict, kernel_size=(1, 1)).to(DEVICE)
        # 第一种
        self.x2_linear = Linear_self(3, 4).to(DEVICE)
        self.final_linear_1 = Linear_self(8, 50).to(DEVICE)
        self.final_linear_2 = Linear_self(10, 1).to(DEVICE)
        # 第三种种
        self.Infin_linear = Linear_self(2, 50).to(DEVICE)
        self.Outfin_linear = Linear_self(50, 1).to(DEVICE)
        self.Feat_lineat = Linear_self(4,1).to(DEVICE)
        # self.Outfin_linear = Linear_self(time_strides, num_for_predict).to(DEVICE)
        self.time_strides = time_strides
        self.vcapLay = VcapModel(1, 1)
        self.DEVICE = DEVICE
        self.to(DEVICE)

    def forward(self, DEVICE, x):
        '''
        :param x: (B, N_nodes, F_in, T_in)
        :return: (B, N_nodes, F_out(1),T_out)
        '''
    # 有时间卷积
      # 第一种 LSTM直接输出F(1)，vcap输出F(1) 之后cat，经过两层LL再Conv2d：效果很差
      # modelname:GL-F(1)-vcap(1)-LL-Conv   
         
        x1 = x[:, :, 0, :].unsqueeze(2)
        x2 = x[:, :, 1:3, :]
        for block in self.BlockList:
            x1 = block(DEVICE, x1)  # b n f(1) T(4)
        x2 = self.vcapLay(x2) # b n f(1) T(4)
        # 将vcap和case特征拼接、经过两个线性层变换
        x_concat = torch.cat([x1, x2], dim=2)  # b n f(2) T(4)
        x2_linear = self.Infin_linear(x_concat)  # b n f(50) T(4)
        x2_linear = F.relu(x2_linear)
        x2_linear = self.Outfin_linear(x2_linear)  # b n f(1) T
        # b n f(1) T(1) ---> b,N,T(1)
        output = self.final_conv(x2_linear.permute(0,3,1,2)).permute(0,2,3,1)[:,:,:,-1]
        return output
         

        # 第二种 LSTM直接输出F(4)再经过L变成F(1)，vcap输出F(1) 之后cat，经过两层LL再Conv2d:效果不错 0.56
        # modelname:GL-F(4)-L-vcap(1)-LL-Conv 
        '''
        x1 = x[:, :, 0, :].unsqueeze(2)
        x2 = x[:, :, 1:3, :]
        for block in self.BlockList:
            x1 = block(DEVICE, x1)  # b n f(4) T(4)
        x1 = self.Feat_lineat(x1)
        x2 = self.vcapLay(x2)  # b n f(1) T(4)
        # 将vcap和case特征拼接、经过两个线性层变换
        x_concat = torch.cat([x1, x2], dim=2)  # b n f(2) T(4)
        x2_linear = self.Infin_linear(x_concat)  # b n f(50) T(4)
        x2_linear = F.relu(x2_linear)
        x2_linear = self.Outfin_linear(x2_linear)  # b n f(1) T(4)
        # b n f(1) T(1) ---> b,N,T(1)
        output = self.final_conv(x2_linear.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)[:, :, :, -1]
        return output
        '''

        # 第三种 将F变成4再变成1，vcap变成1,T变成1，cat后再LL：效果一般
        '''
        x1 = x[:, :, 0, :].unsqueeze(2)
        x2 = x[:, :, 1:3, :]
        for block in self.BlockList:
            x1 = block(DEVICE, x1)  # b n f(4) T(4)
        x1 = self.Feat_lineat(x1)  # b n f(1) T(4)
        x1 = x1[:,:,:,-1].unsqueeze(3) # b n f(1) T(1)
        x2 = self.vcapLay(x2)[:,:,:,-1].unsqueeze(3)  # b n f(1) T(1)
        # 将vcap和case特征拼接、经过两个线性层变换
        x_concat = torch.cat([x1, x2], dim=2)  # b n f(2) T(1)
        x2_linear = self.Infin_linear(x_concat)  # b n f(50) T(1)
        x2_linear = F.relu(x2_linear)
        x2_linear = self.Outfin_linear(x2_linear)  # b n f(1) T
        # b n f(1) T(1) ---> b,N,T(1)
        output = x2_linear[:,:,:,-1]
        # output = self.final_conv(x2_linear.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)[:, :, :, -1]
        return output
        '''


# 不含有时间卷积的模型
class No_TimeCon(nn.Module):
    def __init__(self, DEVICE, nb_block, in_channels, hid_feature, out_feature, nheads, lstm_out, dropout, alpha,
                 adj_mx, time_strides, num_for_predict):
        super(No_TimeCon, self).__init__()
        self.BlockList = nn.ModuleList(
            [ASTGCN_block(DEVICE, in_channels, hid_feature, out_feature, nheads, lstm_out, dropout, alpha, adj_mx)]).to(
            DEVICE)
        self.BlockList.extend(
            [ASTGCN_block(DEVICE, lstm_out, 8, 32, nheads, 4, dropout, alpha, adj_mx) for _ in range(nb_block - 1)]).to(
            DEVICE)
        # (b,N,F_out,T)
        self.final_conv = nn.Conv2d(int(time_strides), num_for_predict, kernel_size=(1, 1)).to(DEVICE)
        # 第一种
        self.x2_linear = Linear_self(3, 4).to(DEVICE)
        self.final_linear_1 = Linear_self(8, 10).to(DEVICE)
        self.final_linear_2 = Linear_self(10, 1).to(DEVICE)
        # 第三种种
        self.Infin_linear = Linear_self(2, 10).to(DEVICE)
        self.Outfin_linear = Linear_self(10, 1).to(DEVICE)
        self.Feat_lineat = Linear_self(4,1).to(DEVICE)
        self.Feat2_lineat = Linear_self(1,4).to(DEVICE)
        # self.Outfin_linear = Linear_self(time_strides, num_for_predict).to(DEVICE)
        self.time_strides = time_strides
        self.vcapLay = VcapModel(1, 1)
        self.DEVICE = DEVICE
        self.to(DEVICE)

    def forward(self, DEVICE, x):
        '''
        :param x: (B, N_nodes, F_in, T_in)
        :return: (B, N_nodes, F_out(1),T_out)
        '''

  # 第一种 LSTM直接输出F(1)，vcap输出F(1) cat，经过两层LL再：效果很差 ==1.017==
  # modelname:GL-F(1)-vcap(1)-LL
  
        ''' 
        x1 = x[:, :, 0, :].unsqueeze(2)
        x2 = x[:, :, 1:3, :]
        for block in self.BlockList:
            x1 = block(DEVICE, x1)  # b n f(1) T(4)
        x2 = self.vcapLay(x2) # b n f(1) T(4)
        # 将vcap和case特征拼接、经过两个线性层变换
        x_concat = torch.cat([x1, x2], dim=2)  # b n f(2) T(4)
        x2_linear = self.Infin_linear(x_concat)  # b n f(50) T(4)
        x2_linear = F.relu(x2_linear)
        x2_linear = self.Outfin_linear(x2_linear)  # b n f(1) T
        # b n f(1) T(4) ---> b,N,T(1)
        output = x2_linear[:,:,:,-1]
        return output
        '''

  # 第二种 LSTM直接输出F(4)再经过L变成F(1)，vcap输出F(1) cat，经过两层LL：
  # modelname:GL-F(4)-L-vcap(1)-LL   
        '''
        x1 = x[:, :, 0, :].unsqueeze(2)
        x2 = x[:, :, 1:3, :]
        for block in self.BlockList:
            x1 = block(DEVICE, x1)  # b n f(4) T(4)
        x1 = self.Feat_lineat(x1)
        x2 = self.vcapLay(x2)  # b n f(1) T(4)
        # 将vcap和case特征拼接、经过两个线性层变换
        x_concat = torch.cat([x1, x2], dim=2)  # b n f(2) T(4)
        x2_linear = self.Infin_linear(x_concat)  # b n f(50) T(4)
        # x2_linear = F.relu(x2_linear)
        x2_linear = self.Outfin_linear(x2_linear)  # b n f(1) T(4)
        # b n f(1) T(1) ---> b,N,T(1)
        output = x2_linear[:,:,:,-1]
        return output
        '''

        # 第三种 LSTM直接输出F(4))，vcap经过L输出F(4) 之后cat，经过两层LL取最后一个时间:
        # modelname:GL-F(4)-vcap(1)-L-cat-LL
         
        x1 = x[:, :, 0, :].unsqueeze(2)
        x2 = x[:, :, 1:3, :]
        for block in self.BlockList:
            x1 = block(DEVICE, x1)  # b n f(4) T(4)
        x2 = self.vcapLay(x2)  # b n f(1) T(1)
        x2 = self.Feat2_lineat(x2)
        # 将vcap和case特征拼接、经过两个线性层变换
        x_concat = torch.cat([x1, x2], dim=2)  # b n f(2) T(1)
        x2_linear = self.Infin_linear(x_concat)  # b n f(50) T(1)
        x2_linear = F.relu(x2_linear)
        x2_linear = self.Outfin_linear(x2_linear)  # b n f(1) T
        # b n f(1) T(1) ---> b,N,T(1)
        output = x2_linear[:,:,:,-1]
        # output = self.final_conv(x2_linear.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)[:, :, :, -1]
        return output
         

# 只有GAT和LSTM
class GAT_LSTM(nn.Module):
    def __init__(self, DEVICE, nb_block, in_channels, hid_feature, out_feature, nheads, lstm_out, dropout, alpha,
                 adj_mx, time_strides, num_for_predict):
        super(GAT_LSTM, self).__init__()
        self.BlockList = nn.ModuleList(
            [ASTGCN_block(DEVICE, in_channels, hid_feature, out_feature, nheads, lstm_out, dropout, alpha, adj_mx)]).to(
            DEVICE)
        self.BlockList.extend(
            [ASTGCN_block(DEVICE, lstm_out, 8, 32, nheads, 4, dropout, alpha, adj_mx) for _ in range(nb_block - 1)]).to(
            DEVICE)
        # (b,N,F_out,T)
        self.final_conv = nn.Conv2d(int(time_strides), num_for_predict, kernel_size=(1, 1)).to(DEVICE)
        # 第一种

        # 第二种
        self.x2_linear = Linear_self(3, 4).to(DEVICE)
        self.final_linear_1 = Linear_self(8, 10).to(DEVICE)
        self.final_linear_2 = Linear_self(10, 1).to(DEVICE)
        # 第三种种
        self.ser_dep = series_decomp(9)
        self.Infin_linear = Linear_self(2, 10).to(DEVICE)
        self.Outfin_linear = Linear_self(10, 1).to(DEVICE)
        self.Feat_lineat = Linear_self(4,1).to(DEVICE)

        # self.Outfin_linear = Linear_self(time_strides, num_for_predict).to(DEVICE)
        self.time_strides = time_strides
        self.vcapLay = VcapModel(1, 1)
        self.DEVICE = DEVICE
        self.to(DEVICE)

    def forward(self, DEVICE, x):
        '''
        :param x: (B, N_nodes, F_in, T_in)
        :return: (B, N_nodes, F_out(1),T_out)
        '''
      # 第一种 输出F(4)经过L变为F(1)，并且取最后一个时间片的值：0.5804
      # modelname:GAT-LSTM-F(4)-L 
         
        x1 = x[:, :, 0, :].unsqueeze(2)
        for block in self.BlockList:
            x1 = block(DEVICE, x1)  # b n f(1) T(4)
        x1 = self.Feat_lineat(x1)
        # b n f(1) T(1) ---> b,N,T(1)
        output = x1[:,:,:,-1]
        return output

        
      # 第二种 LSTM直接输出F(1)，经过一层L输出：效果很差
      # modelname:GAT-LSTM-F(1)-L 
        '''
        x1 = x[:, :, 0, :].unsqueeze(2)
        for block in self.BlockList:
            x1 = block(DEVICE, x1)  # b n f(1) T(4)
        x1 = x1[:,:,:,-1]
        output = x1
        # x2_linear = self.Infin_linear(x_concat)  # b n f(50) T(4)
        # x2_linear = F.relu(x2_linear)
        # x2_linear = self.Outfin_linear(x2_linear)  # b n f(1) T
        # b n f(1) T(1) ---> b,N,T(1)
        return output
        '''

        # 第三种 LSTM直接输出F输出F(1)，卷积取一个时间片：
        # modelname:GAT-LSTM-F(1)-Conv 
        ''' 
        x1 = x[:, :, 0, :].unsqueeze(2)
        for block in self.BlockList:
            x1 = block(DEVICE, x1)  # b n f(4) T(4)
        # b n f(1) T(4) ---> b,N,T(1)
        output = self.final_conv(x1.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)[:, :, :, -1]
        return output
        ''' 

        # 第四种 LSTM直接输出F(4)，经过L变为F(1)，卷积取一个时间片d：0.55848-0.58732
        # modelname:GAT-LSTM-F(4)-L-Conv
        '''
        x1 = x[:, :, 0, :].unsqueeze(2)
        for block in self.BlockList:
            x1 = block(DEVICE, x1)  # b n f(4) T(4)
        x1 = self.Feat_lineat(x1)
        # b n f(1) T(4) ---> b,N,T(1)
        output = self.final_conv(x1.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)[:, :, :, -1]
        return output
        '''
class GAT_LSTM_Vcap(nn.Module):
    def __init__(self, DEVICE, nb_block, in_channels, hid_feature, out_feature, nheads, lstm_out, dropout, alpha,
                 adj_mx, time_strides, num_for_predict):
        super(GAT_LSTM_Vcap, self).__init__()
        self.BlockList = nn.ModuleList(  # GAT_out_feature
            [ASTGCN_block(DEVICE, in_channels, hid_feature, out_feature, nheads, lstm_out, dropout, alpha, adj_mx)]).to(
            DEVICE)
        self.BlockList.extend(
            [ASTGCN_block(DEVICE, lstm_out, 8, 2*out_feature, nheads, 8, dropout, alpha, adj_mx) for _ in range(nb_block - 1)]).to(
            DEVICE)
        # (b,N,F_out,T)
        self.final_conv = nn.Conv2d(int(time_strides), num_for_predict, kernel_size=(1, 1)).to(DEVICE)
        # 第一种
        self.Infin_linear = Linear_self(9, 10).to(DEVICE)
        self.Outfin_linear = Linear_self(10, 1).to(DEVICE)
        self.Feat_lineat = Linear_self(2, 1).to(DEVICE)
        # self.Outfin_linear = Linear_self(time_strides, num_for_predict).to(DEVICE)
        self.time_strides = time_strides
        self.vcapLay = VcapModel(1, 1)
        self.DEVICE = DEVICE
        self.to(DEVICE)

    def forward(self, DEVICE, x):
        '''
        :param x: (B, N_nodes, F_in, T_in)
        :return: (B, N_nodes, F_out(1),T_out)
        '''

        # 第一种 F(4) F(1)-cat-LL
        x1 = x[:, :, 0, :].unsqueeze(2)
        x2 = x[:, :, 1:3, :]
        for block in self.BlockList:
            x1 = block(DEVICE, x1)  # b n f(4) T(4)
        x2 = self.vcapLay(x2)
        x_concat = torch.cat([x1, x2], dim=2)  # b n f(2) T(T)
        # output = self.Feat_lineat(x_concat)
        output = self.Infin_linear(x_concat)  # b n f(1) T(4)
        output = F.relu(output)
        output = self.Outfin_linear(output)  # b n f(1) T(4)
        output = output[:, :, :, -1]
        return output


# 只有Vcap和LL线性
class LL_Vcap(nn.Module):
    def __init__(self, DEVICE, nb_block, in_channels, hid_feature, out_feature, nheads, lstm_out, dropout, alpha,
                 adj_mx, time_strides, num_for_predict):
        super(LL_Vcap, self).__init__()
        self.BlockList = nn.ModuleList(  # GAT_out_feature
            [ASTGCN_block(DEVICE, in_channels, hid_feature, out_feature, nheads, lstm_out, dropout, alpha, adj_mx)]).to(
            DEVICE)
        self.BlockList.extend(
            [ASTGCN_block(DEVICE, lstm_out, 8, 32, nheads, 1, dropout, alpha, adj_mx) for _ in range(nb_block - 1)]).to(
            DEVICE)
        # (b,N,F_out,T)
        self.final_conv = nn.Conv2d(int(time_strides), num_for_predict, kernel_size=(1, 1)).to(DEVICE)
        # 第一种

        # 第二种
        # 第三种种
        self.Infin_linear = Linear_self(2, 10).to(DEVICE)
        self.Outfin_linear = Linear_self(10, 1).to(DEVICE)
        self.Feat_lineat = Linear_self(2, 1).to(DEVICE)
        # self.Outfin_linear = Linear_self(time_strides, num_for_predict).to(DEVICE)
        self.time_strides = time_strides
        self.vcapLay = VcapModel(1, 1)
        self.DEVICE = DEVICE
        self.to(DEVICE)

    def forward(self, DEVICE, x):
        '''
        :param x: (B, N_nodes, F_in, T_in)
        :return: (B, N_nodes, F_out(1),T_out)
        '''
        # 有时间卷积

        # 第一种 x1和vcap cat 经过L输出：效果一般0.61-0.64

        x1 = x[:, :, 0, :].unsqueeze(2)
        x2 = x[:, :, 1:3, :]
        x2 = self.vcapLay(x2)
        x_concat = torch.cat([x1, x2], dim=2)  # b n f(2) T(T)
        # output = self.Feat_lineat(x_concat)
        output = self.Infin_linear(x_concat)  # b n f(1) T(4)
        output = F.relu(output)
        output = self.Outfin_linear(output)  # b n f(1) T(4)
        output = output[:, :, :, -1]
        return output



def make_model(DEVICE, nb_block, in_channels, hid_feature, out_feature, nheads, lstm_out, dropout, alpha, adj_mx,
               time_strides, num_for_predict):
    # model = ASTGCN_TimeCon(DEVICE, nb_block, in_channels, hid_feature, out_feature, nheads, lstm_out, dropout, alpha,
    #                          adj_mx, time_strides, num_for_predict)

    # model = No_TimeCon(DEVICE, nb_block, in_channels, hid_feature, out_feature, nheads, lstm_out, dropout, alpha,
                            #  adj_mx, time_strides, num_for_predict)     
                                              
    model = GAT_LSTM(DEVICE, nb_block, in_channels, hid_feature, out_feature, nheads, lstm_out, dropout, alpha,
                             adj_mx, time_strides, num_for_predict)   
                                          
    # model = GAT_LSTM_Vcap(DEVICE, nb_block, in_channels, hid_feature, out_feature, nheads, lstm_out, dropout, alpha,
                            #  adj_mx, time_strides, num_for_predict)   

    # model = LL_Vcap(DEVICE, nb_block, in_channels, hid_feature, out_feature, nheads, lstm_out, dropout, alpha,
    #                          adj_mx, time_strides, num_for_predict)  

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
        else:
            nn.init.uniform_(p)

    return model
