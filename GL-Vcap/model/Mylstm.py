import numpy as np
import torch
from torch import nn


from lib.data_splite import split_sequences
class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self,DEVICE, in_features, out_features, dropout=0.5, alpha=0.2, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)

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
        # outputs = []
        outputs = torch.empty(4,64,49,1)
        outputs = outputs.cuda()
        (batch_size, num_of_vertices, num_of_features, num_of_timesteps) = x.shape  #(B,N,F,T)
        for time_step in range(num_of_timesteps):
            graph_signal = x[:, :, :, time_step]  # shape is (batch_size, N, F)
            outputs_batch = []
            for i in range(batch_size):
                h = graph_signal[i, :, :]  # shape is (batch_size, N, F)
                Wh = torch.mm(h, self.W)  # h.shape: (N, in_features), Wh.shape: (N, out_features)
                e = self._prepare_attentional_mechanism_input(Wh) #n*N
                e_sof = F.softmax(e,dim=1)
                e1 = e_sof.detach().cpu().numpy()
                # np.savetxt("D:/MasterStudents/2020/LvZhuanghu/ex_result/ASTGCN/e-soft" + appendix + ".txt",
                #            e1)
                zero_vec = -9e15*torch.ones_like(e)# n*n
                adj = adj.cuda()
                ad = adj.detach().cpu().numpy()
                attention = torch.where(adj > 0, e, zero_vec)
                atb = attention.detach().cpu().numpy()
                # np.savetxt("D:/MasterStudents/2020/LvZhuanghu/ex_result/ASTGCN/att-no-soft" + appendix + ".txt",
                #            atb)
                attention = F.softmax(attention, dim=1)
                atsof = attention.cpu().detach().numpy()
                # np.savetxt("D:/MasterStudents/2020/LvZhuanghu/ex_result/ASTGCN/att-soft"+appendix +".txt",
                #            atsof)

                attention = F.dropout(attention, self.dropout, training=self.training)
                atdro = attention.cpu().detach().numpy()
                # np.savetxt("D:/MasterStudents/2020/LvZhuanghu/ex_result/ASTGCN/att-soft -drop" + appendix + ".txt",
                #            atdro)
                h_prime = torch.matmul(attention, Wh)
                outputs_batch.append(torch.unsqueeze(h_prime, 0))  # ?0?????(1,N,F)
            # outputs_batch->(B,N,F)
            outputs_batch_ten = torch.stack(outputs_batch, 0)
            outputs_batch_ten = outputs_batch_ten.permute(0,3,2,1).permute(0,2,1,3)
            outputs = outputs_batch_ten
            # outputs = torch.cat((outputs,outputs_batch_ten),0)
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


class LSTMTest(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.num_directions = 1 # ??LSTM

        self.lstmq = nn.LSTM(self.input_size, self.output_size, self.num_layers, batch_first=True)
        # self.linear = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input_seq):
        # input_seq = input_seq.permute(3,1,2,0)
        #input_seq :(B,N,F,T)
        (batch_size, num_of_vertices, num_of_features,num_of_timesteps ) = input_seq.shape
        target = []
        for time_step in range(batch_size):
            input_ = input_seq[time_step, :, :, :]  # (B,N,F,T)-->(N,F,T)
            input_ = input_.permute(0,2,1)
            # input_ = input_.astype(np.float64)
            h_0 = torch.randn(self.num_directions * self.num_layers, num_of_vertices, self.output_size)
            c_0 = torch.randn(self.num_directions * self.num_layers, num_of_vertices, self.output_size)
            seq_len = input_.shape[1] # (5, 24)
            # input(batch_size, seq_len, input_size)
            # input_seq = input_seq.view(self.batch_size, seq_len, -1)  #
            # output(batch_size, seq_len, num_directions * hidden_size)
            d_0 = (h_0, c_0)
            output, _ = self.lstmq(torch.tensor(input_,dtype=torch.float), d_0) #  (N,T,F)
            # output = output.contiguous().view(batch_size * seq_len, self.output_size) # (T,N,F_out)
            # pred = self.linear(output) # pred(150, 1)
            # pred = output.view(batch_size, seq_len, -1) #
            # pred = pred[:, -1, :]  # (5, 1)
            output = output.permute(0,2,1) # (N,F_out,T)
            target.append(output)
        return  torch.stack(target,0)  #    ?????(B,N,F,T)



class LSTM_for_mls(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, batch_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.num_directions = 1 # ??LSTM
        self.batch_size = batch_size
        self.lstmq = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
        self.linear = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input_seq):
        (batch_size, num_of_vertices, num_of_features, num_of_timesteps) = input_seq.shape  # (B,N,F,T)
        target = []
        for time_step in range(num_of_timesteps):
            input_ = input_seq[:,:,:,time_step] # (B,N,F,T)
            # input_ = input_.astype(np.float64)
            h_0 = torch.randn(self.num_directions * self.num_layers, self.batch_size, self.hidden_size)
            c_0 = torch.randn(self.num_directions * self.num_layers, self.batch_size, self.hidden_size)
            seq_len = input_.shape[1] # (5, 24)
            # input(batch_size, seq_len, input_size)
            # input_seq = input_seq.view(self.batch_size, seq_len, -1)  # (5, 24, 1)
            # output(batch_size, seq_len, num_directions * hidden_size)
            d_0 = (h_0, c_0)
            output, _ = self.lstmq(torch.tensor(input_,dtype=torch.float), d_0) # output(5, 24, 64)
            output = output.contiguous().view(self.batch_size * seq_len, self.hidden_size) # (5 * 24, 64)
            pred = self.linear(output) # pred(150, 1)
            pred = pred.view(self.batch_size, seq_len, -1) # (5, 24, 1)
            # pred = pred[:, -1, :]  # (5, 1)
            target.append(pred)
        return  torch.stack(target,0)
