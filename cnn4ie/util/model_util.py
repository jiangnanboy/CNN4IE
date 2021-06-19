import torch.nn as nn

def init_weights(model):
    '''
    init model weights
    :param model:
    :return:
    '''
    for name,param in model.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)

def epoch_time(start_time, end_time):
    '''
    time-consuming of every epoch
    :param start_time:
    :param end_time:
    :return:
    '''
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs