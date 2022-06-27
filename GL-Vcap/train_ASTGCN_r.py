#!/usr/bin/env python
# coding: utf-8
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from time import time
import shutil
import argparse
import torch.nn.functional as F
import configparser
import wandb
from model.ASTGCN_r import make_model
from lib.utils import load_graphdata_channel1, get_adjacency_matrix, compute_val_loss_mstgcn, \
    predict_and_save_results_mstgcn
from tensorboardX import SummaryWriter
import datetime

parser = argparse.ArgumentParser()
parser.add_argument("--config", default='configurations/PEMS04_astgcn.conf', type=str,
                    help="configuration file path")
args = parser.parse_args()
config = configparser.ConfigParser()
print('Read configuration file: %s' % (args.config))
config.read(args.config)
data_config = config['Data']
training_config = config['Training']

adj_filename = data_config['adj_filename']
graph_signal_matrix_filename = data_config['graph_signal_matrix_filename']
if config.has_option('Data', 'id_filename'):
    id_filename = data_config['id_filename']
else:
    id_filename = None

num_of_vertices = int(data_config['num_of_vertices'])
points_per_hour = int(data_config['points_per_hour'])
num_for_predict = int(data_config['num_for_predict'])
len_input = int(data_config['len_input'])
dataset_name = data_config['dataset_name']

model_name = training_config['model_name']

ctx = training_config['ctx']
os.environ["CUDA_VISIBLE_DEVICES"] = ctx
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# DEVICE = torch.device('cuda:0')
print("CUDA:", USE_CUDA, DEVICE)

learning_rate = float(training_config['learning_rate'])
epochs = int(training_config['epochs'])
start_epoch = int(training_config['start_epoch'])
batch_size = int(training_config['batch_size'])
num_of_weeks = int(training_config['num_of_weeks'])
num_of_days = int(training_config['num_of_days'])
num_of_hours = int(training_config['num_of_hours'])
time_strides = num_of_hours  # strides:shijianbuc
nb_chev_filter = int(training_config['nb_chev_filter'])
nb_time_filter = int(training_config['nb_time_filter'])
in_channels = int(training_config['in_channels'])
nb_block = int(training_config['nb_block'])
K = int(training_config['K'])

folder_dir = '%s｜window:%d｜pre_len:%d｜%s' % (
    model_name, num_of_hours,num_for_predict , learning_rate)
print('folder_dir:', folder_dir)
params_path = os.path.join('experiments', dataset_name, folder_dir)
print('params_path:', params_path)
# (B,N,F,T')

train_loader, train_target_tensor, test_loader, test_target_tensor, _mean, _std = load_graphdata_channel1(
    graph_signal_matrix_filename, num_of_hours,
    num_of_days, num_of_weeks, DEVICE, batch_size)
adj_mx = np.load('data/ajtest.npy')
wandb.init(project="GL-Vcap-6-7", entity="wannalv")
wandb.config={"epochs": epochs, "batch_size": batch_size}
net = make_model(DEVICE, nb_block, in_channels, 4, 16, 8, 32, 0.5, 0.2, adj_mx, time_strides,num_for_predict)


def train_main():
    if (start_epoch == 0) and (not os.path.exists(params_path)):
        os.makedirs(params_path)
        print('create params directory %s' % (params_path))
    elif (start_epoch == 0) and (os.path.exists(params_path)):
        shutil.rmtree(params_path)
        os.makedirs(params_path)
        print('delete the old one and create params directory %s' % (params_path))
    elif (start_epoch > 0) and (os.path.exists(params_path)):
        print('train from params directory %s' % (params_path))
    else:
        raise SystemExit('Wrong type of model!')

    print('param list:')
    print('CUDA\t', DEVICE)
    print('in_channels\t', in_channels)
    print('nb_block\t', nb_block)
    print('nb_chev_filter\t', nb_chev_filter)
    print('nb_time_filter\t', nb_time_filter)
    print('time_strides\t', time_strides)
    print('batch_size\t', batch_size)
    print('graph_signal_matrix_filename\t', graph_signal_matrix_filename)
    print('start_epoch\t', start_epoch)
    print('epochs\t', epochs)

    # criterion = nn.MSELoss().to(DEVICE)
    criterion = F.l1_loss
    # criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    sw = SummaryWriter(logdir=params_path, flush_secs=5)
    print(net)

    print('Net\'s state_dict:')
    total_param = 0
    for param_tensor in net.state_dict():
        print(param_tensor, '\t', net.state_dict()[param_tensor].size())
        total_param += np.prod(net.state_dict()[param_tensor].size())
    print('Net\'s total params:', total_param)

    print('Optimizer\'s state_dict:')
    for var_name in optimizer.state_dict():
        print(var_name, '\t', optimizer.state_dict()[var_name])

    global_step = 0
    best_epoch = 0
    best_val_loss = np.inf

    start_time = time()

    if start_epoch > 0:
        params_filename = os.path.join(params_path, 'epoch_%s.params' % start_epoch)

        net.load_state_dict(torch.load(params_filename))

        print('start epoch:', start_epoch)

        print('load weight from: ', params_filename)
    now = datetime.datetime.now()
    appendix = now.strftime("%m%d-%H%M%S")

    # train model
    for epoch in range(start_epoch, epochs):
        # print("epoch***********",epoch)
        # params_filename = os.path.join(params_path, 'epoch_%s.params' % epoch)
        #
        # val_loss = compute_val_loss_mstgcn(net, val_loader, criterion, sw, epoch)
        #
        # if val_loss < best_val_loss:
        #     best_val_loss = val_loss
        #     best_epoch = epoch
        #     torch.save(net.state_dict(), params_filename)
        #     print('save parameters to file: %s' % params_filename)
        print("epoch***********", epoch)
        net.train()  # ensure dropoutc layers are in train mode
        for batch_index, batch_data in enumerate(train_loader):
            # (B,N,F,T')
            encoder_inputs, labels = batch_data
            optimizer.zero_grad()
            # print("encoder_inputs:shape:",encoder_inputs.shape)
            outputs = net(DEVICE,encoder_inputs)  # (B,N,F_out,T)
            loss = criterion(outputs, labels)
            params_filename = os.path.join(params_path, 'epoch_%s.params' % epoch)
            #
            # train_loss = compute_val_loss_mstgcn(net, val_loader, criterion, sw, epoch)
            #
            if loss < best_val_loss:
                best_val_loss = loss
                best_epoch = epoch
                torch.save(net.state_dict(), params_filename)
                print('save parameters to file: %s' % params_filename)
            loss.backward()
            # print("best:",best_epoch)

            optimizer.step()

            training_loss = loss.item()
            global_step += 1
            sw.add_scalar('training_loss', training_loss, global_step)

            if global_step % 1000 == 0:
                print('global step: %s, training loss: %.2f, time: %.2fs' % (
                    global_step, training_loss, time() - start_time))
        print('Epoch %d, Loss %.3f,' % (epoch, loss,))

        wandb.log({"loss": loss,
                   "learning_rate": learning_rate,
                   "epoch": epochs,
                   "best_epoch": best_epoch,
                   "nb_chev_filter": nb_chev_filter,
                   "batch_size": batch_size,
                   "len_input":len_input,
                   "window_size":num_of_hours,
                   "nb_time_filter":nb_time_filter,
                   "nb_block":nb_block,
                   "num_for_predict":num_for_predict,
                   "model_name":model_name
                   })
    # print('best epoch:', best_epoch)
    # apply the best model on the test set
    predict_main(best_epoch, test_loader, test_target_tensor, _mean, _std, 'test')


def predict_main(global_step, data_loader, data_target_tensor, _mean, _std, type):
    '''

    :param global_step: int
    :param data_loader: torch.utils.data.utils.DataLoader
    :param data_target_tensor: tensor
    :param mean: (1, 1, 3, 1)
    :param std: (1, 1, 3, 1)
    :param type: string
    :return:
    '''

    params_filename = os.path.join(params_path, 'epoch_%s.params' % global_step)
    print('load weight from:', params_filename)

    net.load_state_dict(torch.load(params_filename))

    predict_and_save_results_mstgcn(DEVICE,net, data_loader, data_target_tensor, global_step, _mean, _std, params_path, type)


if __name__ == "__main__":
    train_main()
    exit()
    # predict_main(best_epoch, test_loader, test_target_tensor, _mean, _std, 'test')
    # predict_main(31, test_loader, test_target_tensor, _mean, _std, 'test')
