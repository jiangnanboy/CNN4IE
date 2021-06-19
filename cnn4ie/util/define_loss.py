from torch import nn

def define_loss_ce(PAD_IDX=None):
    '''
    define loss function CE
    :param PAD_IDX:
    :return:
    '''
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    return criterion

def define_loss_bce():
    '''
    define loss function BCE
    :return:
    '''
    criterion = nn.BCELoss()
    return criterion

def define_loss_bcelogits():
    '''
    define loss function BCELogit
    :return:
    '''
    criterion = nn.BCEWithLogitsLoss()
    return criterion