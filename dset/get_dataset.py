from torchtext import data
from torchtext.vocab import Vectors
import pickle
import torch

def build_data_iter(data_catalog,
                    train_file_name,
                    validation_file_name,
                    source_words_path,
                    target_words_path,
                    label_words_path,
                    batch_size,
                    max_length,
                    pretrained_embedding=None):
    '''
    build dataset
    :param data_catalog:
    :param train_file_name:
    :param validation_file_name:
    :param source_words_path:
    :param target_words_path:
    :param label_words_path:
    :param batch_size:
    :return:
    '''
    tokenize = lambda s: s.split()

    SOURCE = data.Field(sequential=True, tokenize=tokenize,
                        lower=True, use_vocab=True,
                        pad_token='<pad>', unk_token='<unk>',
                        batch_first=True, fix_length=max_length,
                        include_lengths=True)  # include_lengths=True为方便之后使用torch的pack_padded_sequence
    
    TARGET = data.Field(sequential=True, tokenize=tokenize,
                        lower=False, use_vocab=True,
                        pad_token='<pad>', unk_token='<unk>',
                        batch_first=True, fix_length=max_length,
                        include_lengths=True)  # include_lengths=True为方便之后使用torch的pack_padded_sequence
    '''
    LABEL = data.Field(
        sequential=False,
        use_vocab=True,
        batch_first=True)
    '''
    
    train, val = data.TabularDataset.splits(
        path=data_catalog,
        skip_header=True,
        train=train_file_name,
        validation=validation_file_name,
        format='csv',
        fields=[('label', None), ('source', SOURCE), ('target', TARGET)])

    if pretrained_embedding:
        vectors = Vectors(name = pretrained_embedding)
        SOURCE.build_vocab(train.source, val.source, vectors = vectors, unk_init = torch.Tensor.normal_) # unk_init对unk词进行初始化
    else:
        SOURCE.build_vocab(train.source, val.source)
    TARGET.build_vocab(train.target, val.target)
    
    print('source vocab len:{}'.format(len(SOURCE.vocab)))
    print('target vocab len:{}'.format(len(TARGET.vocab)))

    #LABEL.build_vocab(train, val)

    # save source words
    with open(source_words_path, 'wb') as f_source_words:
        pickle.dump(SOURCE.vocab, f_source_words)

    # save target words
    with open(target_words_path, 'wb') as f_target_words:
        pickle.dump(TARGET.vocab, f_target_words)

    # save label words
    '''
    with open(label_words_path, 'wb') as f_label_words:
        pickle.dump(LABEL.vocab, f_label_words)
    '''
    
    '''
    train_iter, val_iter = data.Iterator.splits(
        (train, val),
        batch_sizes=(batch_size, len(val)),  # 训练集设置为batch_size,验证集整个集合用于测试
        shuffle=True,
        sort_within_batch=True,  # 为true则一个batch内的数据会按sort_key规则降序排序
        sort_key=lambda x: len(x.source))  # 这里按src的长度降序排序，主要是为后面pack,pad操作)
    '''

    # BucketIterator 把长度相近的文本数据尽量都放到一个batch里
    train_iter, val_iter = data.BucketIterator.splits(
        (train, val),
        batch_sizes=(batch_size, len(val)),
        shuffle=True, # 一般validation不进行shuffle
        sort_within_batch=True,
        sort_key=lambda x: len(x.source))

    if pretrained_embedding:
        return train_iter, val_iter, SOURCE.vocab, len(TARGET.vocab), SOURCE.vocab['<pad>']

    return train_iter, val_iter, len(SOURCE.vocab), len(TARGET.vocab), SOURCE.vocab['<pad>']


