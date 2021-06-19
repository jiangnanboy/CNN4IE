from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_fscore_support, classification_report

def roc_auc_score_binary(y_true, y_pred):
    '''
    binary class
    :param y_true:
    :param y_pred:
    :return:
    '''
    return roc_auc_score(y_true, y_pred)

def roc_auc_score_multi(y_true, y_pred, multi_class='ovo'):
    '''
    multi class
    :param y_true: (batch_size, true_classes)
    :param y_pred: (batch_size, pred_classes)
    :return:
    '''
    return roc_auc_score(y_true, y_pred, multi_class=multi_class)

def p_r_f1(y_true, y_pred, average=None):
    '''
    precision,recall,fscore
    :param y_true:
    :param y_pred:
    :param average:
    :return:
    '''
    return precision_recall_fscore_support(y_true, y_pred, average=average)

def classification_report_f_r_f1(y_true, y_pred, target_names=None):
    '''
    classification report precision, recall, fscore
    :param y_true:
    :param y_pred:
    :param target_names:
    :return:
    '''
    return classification_report(y_true, y_pred, target_names=target_names)
