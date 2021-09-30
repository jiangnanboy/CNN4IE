#import sys
#sys.path.append(r'/home/shiyan/project/CNN4IE/')

import torch
import os
from configparser import ConfigParser
import pickle

from cnn4ie.dscnn.train import Train
from cnn4ie.util.crf_util import get_tags, format_result

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Predict():
    def __init__(self):
        self.model = None
        self.source_vocab = None
        self.target_vocab = None
        self.max_length = 0
        self.tags = list()
        self.tags_map = dict()

    def _predict_ner(self, model, source_vocab, target_vocab, sentence, max_length):
        '''
        predict ner without crf
        :param model:
        :param source_vocab:
        :param target_vocab:
        :param sentence:
        :param max_length
        :return:
        '''
        model.eval()

        tokenized = list(sentence)  # tokenize the sentence
        if len(tokenized) > max_length:
            tokenized = tokenized[:max_length]
        #tokenized = ['<sos>'] + tokenized + ['<eos>']
        indexed = [source_vocab[t] for t in tokenized]  # convert to integer sequence

        #print("tokenized: {}".format(tokenized))
        #print("indexed: {}".format(indexed))

        src_tensor = torch.LongTensor(indexed)  # convert to tensor
        src_tensor = src_tensor.unsqueeze(0).to(DEVICE)  # reshape in form of batch,no. of words

        with torch.no_grad():
            sentence_output = model(src_tensor) # [batch_size, src_len, output_dim]
            pred_token = sentence_output.argmax(2)[:, -1].item()
            pred_token = [target_vocab.itos[i] for i in pred_token]
            return pred_token

    def _load_vocab(self, vocab_path):
        '''
        load vocab
        :param vocab_path:
        :return:
        '''
        if os.path.exists(vocab_path):
            # load vocab
            with open(vocab_path, 'rb') as f_words:
                vocab = pickle.load(f_words)
                return vocab
        else:
            raise FileNotFoundError("File not found!")

    def _predict_crf_ner(self, model, source_vocab, sentence, max_length, tags, tags_map):
        '''
        predict ner with crf
        :param model:
        :param source_vocab:
        :param sentence:
        :param max_length
        :param tags
        :param tags_map
        :return:
        '''

        model.eval()

        tokenized =list(sentence)  # tokenize the sentence
        if len(tokenized) > max_length:
            tokenized = tokenized[:max_length]
        # tokenized = ['<sos>'] + tokenized + ['<eos>']
        indexed = [source_vocab[t] for t in tokenized]  # convert to integer sequence

        #print("tokenized: {}".format(tokenized))
        #print("token index: {}".format(indexed))

        src_tensor = torch.LongTensor(indexed)  # convert to tensor
        src_tensor = src_tensor.unsqueeze(0).to(DEVICE)  # reshape in form of batch,no. of words

        with torch.no_grad():
            predictions = model(src_tensor)

            print('predictions:{}'.format(predictions))
            entities = []
            for tag in tags:
                ner_tags = get_tags(predictions[0], tag, tags_map)
                entities += format_result(ner_tags, sentence, tag)
            return entities

    def load_model_vocab(self, config_path):
        '''
        load model and vocab
        :param config_path:
        :return:
        '''
        if os.path.exists(config_path) and (os.path.split(config_path)[1].split('.')[0] == 'config') and (os.path.splitext(config_path)[1].split('.')[1] == 'cfg'):

            #parent_directory = os.path.dirname(os.path.abspath("."))
            #print('os.getcwd:{}'.format(os.path.dirname(os.path.abspath("."))))
            # load config file
            config = ConfigParser()
            #config.read(os.path.join(os.getcwd(), 'config.cfg'))
            #config.read(os.path.split(config_path)[0], 'config.cfg')
            config.read(config_path)
            section = config.sections()[0]

            # get path, vocabs of source, target
            source_vocab_path = config.get(section, "source_vocab_path")
            # source_vocab_path = os.path.join(os.path.dirname(os.path.abspath('..')), 'data', source_vocab_path)
            print('source_vocab_path:{}'.format(source_vocab_path))

            target_vocab_path = config.get(section, "target_vocab_path")
            # target_vocab_path = os.path.join(os.path.dirname(os.path.abspath('..')), 'data', target_vocab_path)
            print('target_vocab_path:{}'.format(target_vocab_path))

            # load source vocab
            self.source_vocab = self._load_vocab(source_vocab_path)
            print("source_vocab size:{}".format(len(self.source_vocab)))

            # load target vocab
            self.target_vocab = self._load_vocab(target_vocab_path)
            print("target_vocab size:{}".format(len(self.target_vocab)))
            tags = set()
            kv = self.target_vocab.stoi
            for k, v in kv.items():
                self.tags_map[k] = v
                index = k.find('_')
                if index != -1:
                    k = k[index + 1:]
                    tags.add(k)
            self.tags = list(tags)
            print('tags:{}'.format(self.tags))
            print('tags_map:{}'.format(self.tags_map))

            # model save/load path
            model_path = config.get(section, "model_path")
            # model_path = os.path.join(os.path.dirname(os.path.abspath('..')), "model", model_path)

            # model param config
            self.max_length = config.getint(section, "max_length")
            input_dim = len(self.source_vocab)
            output_dim = len(self.target_vocab)
            emb_dim = config.getint(section, "emb_dim")
            hid_dim = config.getint(section, "hid_dim")
            cnn_layers = config.getint(section, "cnn_layers")
            encoder_layers = config.getint(section, "encoder_layers")
            kernel_size = config.getint(section, "kernel_size")
            dropout = config.getfloat(section, "dropout")
            loss_name = config.get(section, 'loss')
            PAD_IDX = self.source_vocab['<pad>']

            # define loss
            if loss_name == 'crf':
                use_crf = True
            else:
                use_crf = False

            # load model
            self.model = Train.load_model(input_dim,
                               output_dim,
                               emb_dim,
                               hid_dim,
                               cnn_layers,
                               encoder_layers,
                               kernel_size,
                               dropout,
                               PAD_IDX,
                               self.max_length,
                               model_path,
                               use_crf=use_crf)
        else:
            raise FileNotFoundError('File config.cfg not found : ' + config_path)

    def predict(self, sentence):
        '''
        predict
        :param sentence:
        :return:
        '''
        if len(sentence.strip()) == 0 or sentence == None:
            raise ValueError('Invalid parameter：' + sentence)

        if self.model.use_crf:
            predictions = self._predict_crf_ner(self.model, self.source_vocab, sentence, self.max_length, self.tags, self.tags_map)
        else:
            predictions = self._predict_ner(self.model, self.source_vocab, self.target_vocab, sentence, self.max_length)

        return predictions

if __name__ == '__main__':
    config_path = os.path.join(os.getcwd(), 'config.cfg')
    predict = Predict()
    predict.load_model_vocab(config_path)
    result = predict.predict('本报北京２月２８日讯记者苏宁报道：八届全国人大常委会第三十次会议今天下午在京闭幕。')
    print('predict result:{}'.format(result))
    # predict result:[{'start': 2, 'stop': 4, 'word': '北京', 'type': 'LOC'}, {'start': 12, 'stop': 14, 'word': '苏宁', 'type': 'LOC'}, {'start': 32, 'stop': 36, 'word': '今天下午', 'type': 'T'}]

