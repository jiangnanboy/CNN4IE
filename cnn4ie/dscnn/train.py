
import torch
import torch.nn.functional as F

import time
import math
import os
from configparser import ConfigParser
import tqdm
import numpy as np

from cnn4ie.dscnn.model import MultiLayerResDSCNN
from dset.get_dataset import build_data_iter
from cnn4ie.util.model_util import init_weights, epoch_time
from cnn4ie.util import define_optimizer
from cnn4ie.util import define_loss
from cnn4ie.util import metrics, crf_util

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Train():
    def __init__(self):
        pass

    def _define_model(self,
                     input_dim,
                     output_dim,
                     emb_dim,
                     hid_dim,
                     cnn_layers,
                     encoder_layers,
                     kernel_size,
                     dropout,
                     PAD_IDX,
                     max_length,
                     pretrained_embedding_vocab=None,
                     init=True,
                     use_crf=True):
        '''
        define model
        :param input_dim:
        :param output_dim
        :param emb_dim:
        :param hid_dim:
        :param cnn_layers:
        :param encoder_layers:
        :param kernel_size:
        :param dropout:
        :param use_crf:
        :param PAD_IDX
        :param pretrained_embedding_vocab
        :param init
        :return:
        '''
        model = MultiLayerResDSCNN(input_dim,
                                 output_dim,
                                 emb_dim,
                                 hid_dim,
                                 cnn_layers,
                                 encoder_layers,
                                 kernel_size,
                                 dropout,
                                 PAD_IDX,
                                 max_length,
                                 use_crf=use_crf)

        # init model weights
        if init:
            model.apply(init_weights)

        # init model token embedding
        if pretrained_embedding_vocab:
            model.tok_embedding.weight.data.copy_(pretrained_embedding_vocab.vectors)
            UNK_IDX = pretrained_embedding_vocab['<unk>']
            # pre-trained weights of the unk and pad word vectors are not trained on our dataset corpus, it is best to set them to zero
            model.tok_embedding.weight.data[UNK_IDX] = torch.zeros(emb_dim)
            model.tok_embedding.weight.data[PAD_IDX] = torch.zeros(emb_dim)

        return model.to(DEVICE)

    @staticmethod
    def load_model(input_dim,
                   output_dim,
                   emb_dim,
                   hid_dim,
                   cnn_layers,
                   encoder_layers,
                   kernel_size,
                   dropout,
                   PAD_IDX,
                   max_length,
                   model_path,
                   use_crf=True):
        '''
        load model
        :param input_dim:
        :param output_dim
        :param emb_dim:
        :param hid_dim:
        :param cnn_layers:
        :param encoder_layers:
        :param kernel_size:
        :param dropout:
        :param PAD_IDX
        :param model_path
        :param use_crf
        :return:
        '''
        model = MultiLayerResDSCNN(input_dim,
                                 output_dim,
                                 emb_dim,
                                 hid_dim,
                                 cnn_layers,
                                 encoder_layers,
                                 kernel_size,
                                 dropout,
                                 PAD_IDX,
                                 max_length,
                                 use_crf=use_crf)
        # load model
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        else:
            raise FileNotFoundError('Not found model file!')

        return model.to(DEVICE)

    def _train(self, model, train_iter, optimizer, criterion, clip):
        '''
        trainning module
        :param model:
        :param iterator:
        :param optimizer:
        :param criterion:
        :param clip:
        :return:
        '''
        model.train()
        epoch_loss = 0
        if model.use_crf:
            for batch in train_iter:
                source, _ = batch.source
                target, _ = batch.target

                source = source.to(DEVICE)
                target = target.to(DEVICE)

                optimizer.zero_grad()

                loss = model.log_likelihood(source, target)

                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

                optimizer.step()

                epoch_loss += loss.item()

            return epoch_loss / len(train_iter)

        else:
            for i, batch in enumerate(train_iter):
                source, _ = batch.source
                target, _ = batch.target

                source = source.to(DEVICE)
                target = target.to(DEVICE)

                optimizer.zero_grad()

                out = model(source)  # [batch_size, src_len, output_dim]
                out = out.view(-1, out.shape[-1]) # [batch_size * src_len, output_dim]

                out = out.contiguous().view(-1, out.shape[-1])  # [batch_size * src_len, output_dim]
                target = target.contiguous().view(-1)  # [batch_size * src_len]

                # loss
                loss = criterion(out, target)

                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

                optimizer.step()

                epoch_loss += loss.item()

            return epoch_loss / len(train_iter)

    def _validate(self, model, val_iter, criterion):
        '''
        validation module
        :param model:
        :param iterator:
        :param criterion:
        :return:
        '''
        model.eval()

        epoch_loss = 0
        if model.use_crf:
            with torch.no_grad():
                preds, labels = [], []
                for batch in val_iter:
                    source, _ = batch.source
                    target, _ = batch.target

                    source = source.to(DEVICE)
                    target = target.to(DEVICE)

                    out = model(source)  # [batch_size, src_len, output_dim]

                    #  the length of non-zero true labels
                    non_zero = []
                    for i in target.cpu():
                        tmp = []
                        for j in i:
                            if j.item() > 0:
                                tmp.append(j.item())
                        non_zero.append(tmp)

                    for index, i in enumerate(out):
                        preds += i[:len(non_zero[index])]

                    for index, i in enumerate(target.tolist()):
                        labels += i[:len(non_zero[index])]

                    

                    # loss
                    loss = model.log_likelihood(source, target)

                    epoch_loss += loss.item()
                # p,r,f1 metrics
                report = metrics.classification_report_f_r_f1(labels, preds)
            return epoch_loss / len(val_iter), report

        else:
            with torch.no_grad():
                labels = np.array([])
                predicts = np.array([])
                for batch in tqdm.tqdm(val_iter):
                    source, _ = batch.source
                    target, _ = batch.target

                    source = source.to(DEVICE)
                    target = target.to(DEVICE)

                    out = model(source)  # [batch_size, src_len, output_dim]
                    out = out.view(-1, out.shape[-1])  # [batch_size * src_len, output_dim]

                    out = out.contiguous().view(-1, out.shape[-1])  # [batch_size * src_len, output_dim]
                    target = target.contiguous().view(-1)  # [batch_size * src_len]

                    # p,r,f1 metrics
                    prediction = torch.max(F.softmax(out, dim=1), dim=1)[1]
                    pred_y = prediction.cpu().data.numpy().squeeze()
                    target_y = target.cpu().data.numpy()
                    labels = np.append(labels, target_y)
                    predicts = np.append(predicts, pred_y)

                    # loss
                    loss = criterion(out, target)

                    epoch_loss += loss.item()
                report = metrics.classification_report_f_r_f1(labels, predicts)
            return epoch_loss / len(val_iter), report

    def _validate_2(self, model, val_iter, criterion, tags, tags_map):
        '''
        validation PER,ORG,LOC,T
        :param model:
        :param val_iter:
        :param criterion:
        :param tags
        :param tags_map
        :return:
        '''

        model.eval()

        epoch_loss = 0
        if model.use_crf:
            with torch.no_grad():
                for batch in val_iter:
                    source, _ = batch.source
                    target, _ = batch.target

                    source = source.to(DEVICE)
                    target = target.to(DEVICE)

                    out = model(source)  # [batch_size, src_len, output_dim]
                    print('\treport:')
                    for tag in tags:
                        crf_util.f1_score(target, out, tag, tags_map)
                    # loss
                    loss = model.log_likelihood(source, target)
                    epoch_loss += loss.item()
            return epoch_loss / len(val_iter)

        else:
            with torch.no_grad():
                for batch in tqdm.tqdm(val_iter):
                    source, _ = batch.source
                    target, _ = batch.target

                    source = source.to(DEVICE)
                    target = target.to(DEVICE)

                    out = model(source)  # [batch_size, src_len, output_dim]
                    print('\treport')
                    for tag in tags:
                        crf_util.f1_score(target, out, tag, tags_map)
                    # loss
                    loss = criterion(out, target)
                    epoch_loss += loss.item()
            return epoch_loss / len(val_iter)

    def _train_val_main(self, model, optimizer, criterion, clip, n_epochs, train_iter, val_iter, model_path):
        '''
        trainning and validation
        :param model:
        :param optimizer:
        :param criterion:
        :param clip:
        :param n_epochs:
        :param train_iter:
        :param val_iter:
        :param model_path:
        :return:
        '''

        best_valid_loss = float('inf')
        # use crf
        if model.use_crf:
            for epoch in range(n_epochs):
                start_time = time.time()

                train_loss = self._train(model, train_iter, optimizer, criterion, clip)
                valid_loss, report = self._validate(model, val_iter, criterion)

                end_time = time.time()
                epoch_mins, epoch_secs = epoch_time(start_time, end_time)

                if valid_loss < best_valid_loss:
                    best_valid_loss = valid_loss
                    torch.save(model.state_dict(), model_path)

                try:
                    train_ppl = math.exp(train_loss)
                except OverflowError:
                    train_ppl = float('inf')

                try:
                    val_ppl = math.exp(valid_loss)
                except OverflowError:
                    val_ppl = float('inf')

                print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
                print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {train_ppl}')
                print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {val_ppl}')
                print(f'\t Val. report: {report}')

        else:
            for epoch in range(n_epochs):
                start_time = time.time()

                train_loss = self._train(model, train_iter, optimizer, criterion, clip)
                valid_loss, report = self._validate(model, val_iter, criterion)

                end_time = time.time()

                epoch_mins, epoch_secs = epoch_time(start_time, end_time)

                if valid_loss < best_valid_loss:
                    best_valid_loss = valid_loss
                    torch.save(model.state_dict(), model_path)

                try:
                    train_ppl = math.exp(train_loss)
                except OverflowError:
                    train_ppl = float('inf')

                try:
                    val_ppl = math.exp(valid_loss)
                except OverflowError:
                    val_ppl = float('inf')

                print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
                print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {train_ppl}')
                print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {val_ppl}')
                print(f'\t Val. report: {report}')

    def train_model(self, config_path):
        if os.path.exists(config_path) and (os.path.split(config_path)[1].split('.')[0] == 'config') and (
                os.path.splitext(config_path)[1].split('.')[1] == 'cfg'):
            # load config file
            config = ConfigParser()
            config.read(config_path)
            section = config.sections()[0]

            # train and val file
            data_catalog = config.get(section, "data_catalog")
            # data_catalog = os.path.join(os.path.dirname(os.path.abspath('..')), data_catalog)
            train_file_name = config.get(section, "train_file_name")
            validation_file_name = config.get(section, "validation_file_name")

            # save vocabs of source, target, label
            source_vocab_path = config.get(section, "source_vocab_path")
            # source_vocab_path = os.path.join(os.path.dirname(os.path.abspath('..')), 'data', source_vocab_path)

            target_vocab_path = config.get(section, "target_vocab_path")
            # target_vocab_path = os.path.join(os.path.dirname(os.path.abspath('..')), 'data', target_vocab_path)

            label_vocab_path = config.get(section, "label_vocab_path")
            # label_vocab_path = os.path.join(os.path.dirname(os.path.abspath('..')), 'data', label_vocab_path)

            pretrained_embedding_path = config.get(section, "pretrained_embedding_path")
            # pretrained_embedding_path = os.path.join(os.path.dirname(os.path.abspath('..')), 'data', pretrained_embedding_path)

            # model save/load path
            model_path = config.get(section, "model_path")
            # model_path = os.path.join(os.path.dirname(os.path.abspath('..')), "model", model_path)

            # model param config
            input_dim = config.getint(section, "input_dim")
            output_dim = config.getint(section, "output_dim")
            emb_dim = config.getint(section, "emb_dim")
            hid_dim = config.getint(section, "hid_dim")
            cnn_layers = config.getint(section, "cnn_layers")
            encoder_layers = config.getint(section, "encoder_layers")
            kernel_size = config.getint(section, "kernel_size")
            dropout = config.getfloat(section, "dropout")
            max_length = config.getint(section, "max_length")

            lr = config.getfloat(section, "lr")
            lr_decay = config.getfloat(section, 'lr_decay')
            weight_decay = config.getfloat(section, "weight_decay")
            gamma = config.getfloat(section, "gamma")
            momentum = config.getfloat(section, "momentum")
            eps = config.getfloat(section, "eps")
            batch_size = config.getint(section, "batch_size")
            clip = config.getfloat(section, "clip")
            n_epochs = config.getint(section, "n_epochs")

            optimizer_name = config.get(section, "optimizer")
            loss_name = config.get(section, "loss")

            pretrained_embedding_vocab = None
            # load pretrained embedding from file
            if os.path.exists(pretrained_embedding_path):
                # get train and val data, source_dict_size_embedding, target dict size, padding_idx
                train_iter, val_iter, pretrained_embedding_vocab, output_dim, PAD_IDX = build_data_iter(data_catalog,
                                                                                                        train_file_name,
                                                                                                        validation_file_name,
                                                                                                        source_vocab_path,
                                                                                                        target_vocab_path,
                                                                                                        label_vocab_path,
                                                                                                        batch_size,
                                                                                                        max_length,
                                                                                                        pretrained_embedding_path)
                input_dim = pretrained_embedding_vocab.vectors.shape[0]
                emb_dim = pretrained_embedding_vocab.vectors.shape[1]

            else:
                # get train and val data, source dict size, target dict size size, padding_idx
                train_iter, val_iter, input_dim, output_dim, PAD_IDX = build_data_iter(data_catalog,
                                                                                       train_file_name,
                                                                                       validation_file_name,
                                                                                       source_vocab_path,
                                                                                       target_vocab_path,
                                                                                       label_vocab_path,
                                                                                       batch_size,
                                                                                       max_length)

            # define loss
            if loss_name == 'crf':
                criterion = None
                use_crf = True

            elif loss_name == 'ce':
                criterion = define_loss.define_loss_ce(PAD_IDX)
                use_crf = False

            elif loss_name == 'bce':
                criterion = define_loss.define_loss_bce()
                use_crf = False

            elif loss_name == 'bcelogits':
                criterion = define_loss.define_loss_bcelogits()
                use_crf = False

            else:
                raise NameError('No define loss function name!')

            print('input_dim:{}'.format(input_dim))
            print('emb_dim:{}'.format(emb_dim))

            # define model
            model = self._define_model(input_dim,
                                 output_dim,
                                 emb_dim,
                                 hid_dim,
                                 cnn_layers,
                                 encoder_layers,
                                 kernel_size,
                                 dropout,
                                 PAD_IDX,
                                 max_length,
                                 pretrained_embedding_vocab,
                                 True,
                                 use_crf)

            # define optimizer
            if optimizer_name == 'adam':
                optimizer = define_optimizer.define_optimizer_adam(model, lr=lr, weight_decay=weight_decay)

            elif optimizer_name == 'adamw':
                optimizer = define_optimizer.define_optimizer_adamw(model, lr=lr, weight_decay=weight_decay)

            elif optimizer_name == 'sgd':
                optimizer = define_optimizer.define_optimizer_sgd(model, lr=lr, momentum=momentum, weight_decay=weight_decay)

            elif optimizer_name == 'adagrad':
                optimizer = define_optimizer.define_optimizer_adagrad(model, lr=lr, lr_decay=lr_decay,
                                                                      weight_decay=weight_decay)

            elif optimizer_name == 'rmsprop':
                optimizer = define_optimizer.define_optimizer_rmsprop(model, lr=lr, weight_decay=weight_decay,
                                                                      momentum=momentum)

            elif optimizer_name == 'adadelta':
                optimizer = define_optimizer.define_optimizer_adadelta(model, lr=lr, weight_decay=weight_decay)

            else:
                raise NameError('No define optimization function name!')

            # train and validate
            self._train_val_main(model, optimizer, criterion, clip, n_epochs, train_iter, val_iter, model_path)

        else:
            raise FileNotFoundError('File config.cfg not found : ' + config_path)

if __name__ == '__main__':
    config_path = os.path.join(os.getcwd(), 'config.cfg')
    train = Train()
    train.train_model(config_path)

