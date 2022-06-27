import torch
import torch.nn as nn
import math
import cmath
import torch.nn.functional as F


class VcapModel(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.W = nn.Parameter(torch.FloatTensor(size=(input_size, output_size)))
        self.A = nn.Parameter(torch.FloatTensor(size=(1, 1)))


    def forward(self, x):
        b, n, f, t = x.shape
        outputs_batch = []
        for batch_step in range(b):
            x_batch = x[batch_step, :, :, :]  # n,f,t
            T = x_batch[:, 0:1, :]  # n,1,t
            R = x_batch[:, 1:2, :]  # n,1,t
            m = 10 * R
            # m = self.W * R
            gpc = 36.5 / (T + 2.0 - 9.9) + 0.5
            a = 0.7 / gpc
            p = 0.5 ** (1 / gpc)
            # p = self.A ** (1 / gpc)
            n = 111.0 / ((2.0 * (26.5 / T + 2.0 - 9.9) / gpc) + T - 18)
            up = -(m * a.pow(2)) * (p ** n)
            down = torch.log(p)
            vcap = up / down  # （N，T）;t个时刻的Vcap值
            outputs_batch.append(vcap)
        outputs_batch_ten = torch.stack(outputs_batch, 0)  # b n f_out t
        return outputs_batch_ten
