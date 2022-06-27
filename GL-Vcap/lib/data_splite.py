#######################################多变量，多步骤预测
from matplotlib import pyplot as plt
from numpy import array



# split a multivariate sequence into samples
def split_sequences(sequences, n_steps_in, n_steps_out,input_list,pred_list):
    X, y = list(), list()
    for i in range(len(sequences)):
        # find the end of this pattern
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out - 1
        # check if we are beyond the dataset
        if out_end_ix +1 > len(sequences):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequences[i:end_ix,:, input_list], sequences[end_ix :out_end_ix+1,:, pred_list]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)





