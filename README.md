# CNN4IE

中文信息抽取工具。使用CNN的不同变体进行信息抽取，以后会持续加入不同模型。该项目使用pytorch，python开发。

**CNN4IE**将各种改进版本的conv进行改动用于中文信息抽取。

**Guide**

- [Intro](#Intro)
- [Model](#Model)
- [Evaluate](#Evaluate)
- [Install](#install)
- [Dataset](#Dataset)
- [Todo](#Todo)
- [Cite](#Cite)
- [Reference](#reference)

## Intro

目前主要实现中文实体抽取：

训练样本以B、I、O形式进行标注。

## Model
### 模型
模型里面的conv块部分主要来自后面的paper中的部分模块。
* 1.MultiLayerResCNN(cnn4ie/mlrescnn)：多层残差CNN(+CRF)， [Convolutional Sequence to Sequence Learning](https://arxiv.org/abs/1705.03122) 。
* 2.MultiLayerResDSCNN(cnn4ie/dscnn)：多层深度可分离depthwise_separable_convolutionCNN(+CRF)， [Xception: Deep Learning with Depthwise Separable Convolutions](https://arxiv.org/pdf/1610.02357.pdf) 。
* 3.MultiLayerAugmentedCNN(cnn4ie/attention_augmented_cnn)：多层注意力增强CNN(+CRF)， [Attention Augmented Convolutional Networks](https://arxiv.org/pdf/1904.09925.pdf) 。
* 4.MultiLayerLambdaCNN(cnn4ie/lambda_cnn)：多层LambdaCNN(+CRF)， [LambdaNetworks: Modeling long-range Interactions without Attention](https://openreview.net/forum?id=xTJEN-ggl1b) 。
* 5.MultiLayerResLWCNN(cnn4ie/lcnn)：多层轻量LightweightCNN(+CRF)， [Pay Less Attention with Lightweight and Dynamic Convolutions](https://arxiv.org/pdf/1901.10430.pdf) 。
* 6.MultiLayerResDYCNN(cnn4ie/dcnn)：多层动态DynamicCNN(+CRF)， [Pay Less Attention with Lightweight and Dynamic Convolutions](https://arxiv.org/pdf/1901.10430.pdf) 。
* 7.MultiLayerStdAttnCNN(cnn4ie/stand_alone_self_attention_cnn)：多层独立自注意力stand_alone_self_attention_CNN(+CRF)，[Stand-Alone Self-Attention in Vision Models](https://arxiv.org/pdf/1906.05909.pdf) 。
* 8.MultiLayerCSAttCNN(cnn4ie/channel_spatial_attention_cnn)，多层联合通道和空间注意力channel_spatial_attention_CNN(+CRF)，[CBAM: Convolutional Block Attention Module](https://arxiv.org/pdf/1807.06521.pdf) 。
* 9.MultiLayerSACNN(cnn4ie/self_attention_cnn)，多层self-attention_CNN(+CRF)，[Self-Attention Generative Adversarial Networks](https://arxiv.org/pdf/1805.08318.pdf) 。
* 10.MultiLayerGroupMixedCNN(cnn4ie/mixed_depthwise_cnn)，多层mixed_depthwise_CNN(+CRF)，[MixConv: Mixed Depthwise Convolutional Kernels](https://arxiv.org/pdf/1907.09595.pdf) 。
* 11.MultiLayerMultiCNN(cnn4ie/multi_cnn)，多层multiconv_CNN(+CRF)，[Character-Level Translation with Self-attention](https://arxiv.org/pdf/2004.14788.pdf) 。
* 12.MultiLayerMixedAttCNN(cnn4ie/mixed_attention_cnn)，多层混合(全局和局部attention)mixed_attention_cnn，[ConvBERT: Improving BERT with Span-based Dynamic Convolution](https://arxiv.org/pdf/2008.02496.pdf) 。

#### Usage
- 相关参数的配置config见每个模型文件夹中的config.cfg文件，训练和预测时会加载此文件。

- 训练及预测(支持加载预训练的embedding向量)

     ##### 1.MultiLayerResCNN(cnn4ie/mlrescnn)
     
     (1).训练
    ```
    from cnn4ie.mlrescnn.train import Train
    train = Train()
    train.train_model('config.cfg')
    ```
  ```
  Epoch: 199 | Time: 0m 4s
	Train Loss: 228.545 | Train PPL: 1.802960293422957e+99
	 Val. Loss: 433.577 |  Val. PPL: 1.9966207577208172e+188
	 Val. report:               precision    recall  f1-score   support

           1       1.00      1.00      1.00      4539
           2       0.98      0.99      0.99      4926
           3       0.90      0.83      0.86       166
           4       0.74      0.98      0.84        52
           5       0.94      0.77      0.84       120
           6       0.76      0.97      0.85        39
           7       0.82      0.87      0.85        54
           8       0.93      0.74      0.82        68
           9       0.95      0.77      0.85        26
          10       1.00      0.80      0.89        10

    accuracy                           0.98     10000
    macro avg       0.90      0.87      0.88     10000
    weighted avg       0.99      0.98      0.98     10000
    ```
      
    (2).预测

    ```
    from cnn4ie.mlrescnn.predict import Predict
  
    predict = Predict()
    predict.load_model_vocab('config_cfg')
    result = predict.predict('据新华社报道，安徽省六安市被评上十大易居城市！')
  
    print(result)
    ```
    ```
    [{'start': 7, 'stop': 13, 'word': '安徽省六安市', 'type': 'LOC'}, {'start': 1, 'stop': 4, 'word': '新华社', 'type': 'ORG'}]
    ```
    ##### 2.MultiLayerResDSCNN(cnn4ie/dscnn)
    
    (1).训练
    ```
    from cnn4ie.dscnn.train import Train
    train = Train()
    train.train_model('config.cfg')
    ```
  ```
  Epoch: 192 | Time: 0m 3s
	Train Loss: 191.273 | Train PPL: 1.172960293422957e+99
	 Val. Loss: 533.260 |  Val. PPL: 5.2866207577208172e+188
	 Val. report:               precision    recall  f1-score   support

           1       0.99      1.00      1.00      4539
           2       0.98      0.98      0.98      4926
           3       0.92      0.82      0.87       166
           4       0.82      0.88      0.85        52
           5       0.84      0.76      0.80       120
           6       0.90      0.95      0.92        39
           7       0.90      0.85      0.88        54
           8       0.84      0.71      0.77        68
           9       0.85      0.65      0.74        26
          10       1.00      0.70      0.82        10

    accuracy                           0.98     10000
    macro avg       0.91      0.83      0.86     10000
    weighted avg       0.98      0.98      0.98     10000
    ```
    (2).预测
    ```
    from cnn4ie.dscnn.predict import Predict
  
    predict = Predict()
    predict.load_model_vocab('config.cfg')
    result = predict.predict('本报北京２月２８日讯记者苏宁报道：八届全国人大常委会第三十次会议今天下午在京闭幕。')
  
    print(result)
    ```
    ```
    [{'start': 2, 'stop': 4, 'word': '北京', 'type': 'LOC'}, {'start': 12, 'stop': 14, 'word': '苏宁', 'type': 'LOC'}, {'start': 32, 'stop': 36, 'word': '今天下午', 'type': 'T'}]
    ```
    ##### 3.MultiLayerAugmentedCNN(cnn4ie/attention_augmented_cnn)
    
    (1).训练
    ```
    from cnn4ie.attention_augmented_cnn.train import Train
    train = Train()
    train.train_model('config.cfg')
    ```
  ```
  Epoch: 192 | Time: 0m 3s
        Train Loss: 185.204 | Train PPL: 2.711303579086953e+80
         Val. Loss: 561.592 |  Val. PPL: 7.877783034926193e+243
         Val. report:               precision    recall  f1-score   support

           1       0.99      1.00      1.00      4539
           2       0.98      0.99      0.98      4926
           3       0.96      0.77      0.85       166
           4       0.81      0.85      0.83        52
           5       0.88      0.71      0.78       120
           6       0.90      0.90      0.90        39
           7       0.90      0.85      0.88        54
           8       0.85      0.69      0.76        68
           9       1.00      0.42      0.59        26
          10       1.00      0.50      0.67        10

    accuracy                           0.98     10000
    macro avg       0.93      0.77      0.82     10000
    weighted avg       0.98      0.98      0.98     10000
    ```
    (2).预测
    ```
    from cnn4ie.attention_augmented_cnn.predict import Predict
  
    predict = Predict()
    predict.load_model_vocab('config.cfg')
    result = predict.predict('本报北京２月２８日讯记者苏宁报道：八届全国人大常委会第三十次会议今天下午在京闭幕。')
  
    print(result)
    ```
    ```
    [{'start': 2, 'stop': 4, 'word': '北京', 'type': 'LOC'}, {'start': 12, 'stop': 14, 'word': '苏宁', 'type': 'LOC'}, {'start': 32, 'stop': 36, 'word': '今天下午', 'type': 'T'}]
    ```
    ##### 4.MultiLayerLambdaCNN(cnn4ie/lambda_cnn)
    
    (1).训练
    ```
    from cnn4ie.lambda_cnn.train import Train
    train = Train()
    train.train_model('config.cfg')
    ```
  ```
  Epoch: 197 | Time: 0m 2s
        Train Loss: 198.344 | Train PPL: 1.3800537707438322e+86
         Val. Loss: 668.780 |  Val. PPL: 2.8022239331403918e+290
         Val. report:               precision    recall  f1-score   support

           1       0.99      1.00      1.00      4539
           2       0.98      0.98      0.98      4926
           3       0.80      0.78      0.79       166
           4       0.89      0.90      0.90        52
           5       0.86      0.77      0.81       120
           6       0.90      0.92      0.91        39
           7       0.81      0.87      0.84        54
           8       0.88      0.75      0.81        68
           9       0.93      0.54      0.68        26
          10       1.00      0.70      0.82        10

    accuracy                           0.98     10000
    macro avg       0.90      0.82      0.85     10000
    weighted avg       0.98      0.98      0.98     10000
    ```
    (2).预测
    ```
    from cnn4ie.lambda_cnn.predict import Predict
  
    predict = Predict()
    predict.load_model_vocab('config.cfg')
    result = predict.predict('本报北京２月２８日讯记者苏宁报道：八届全国人大常委会第三十次会议今天下午在京闭幕。')
  
    print(result)
    ```
    ```
    [{'start': 2, 'stop': 4, 'word': '北京', 'type': 'LOC'}, {'start': 12, 'stop': 14, 'word': '苏宁', 'type': 'LOC'}, {'start': 32, 'stop': 36, 'word': '今天下午', 'type': 'T'}]
    ```
    ##### 5.MultiLayerResLWCNN(cnn4ie/lcnn)
    
    (1).训练
    ```
    from cnn4ie.lcnn.train import Train
    train = Train()
    train.train_model('config.cfg')
    ```
  ```
  Epoch: 190 | Time: 0m 4s
        Train Loss: 195.472 | Train PPL: 7.807223255192846e+84
         Val. Loss: 453.642 |  Val. PPL: 1.0328983269312897e+197
         Val. report:               precision    recall  f1-score   support

           1       0.99      1.00      1.00      5925
           2       0.99      0.98      0.98      5501
           3       0.90      0.85      0.87       174
           4       0.72      0.93      0.81        57
           5       0.92      0.81      0.86       122
           6       0.82      0.91      0.86        44
           7       0.84      0.85      0.85        62
           8       0.92      0.77      0.84        71
           9       0.66      0.81      0.72        31
          10       0.91      0.77      0.83        13

   accuracy                           0.98     12000
   macro avg       0.86      0.87      0.86     12000
   weighted avg       0.98      0.98      0.98     12000
    ```
    (2).预测
    ```
    from cnn4ie.lcnn.predict import Predict
  
    predict = Predict()
    predict.load_model_vocab('config.cfg')
    result = predict.predict('本报北京２月２８日讯记者苏宁报道：八届全国人大常委会第三十次会议今天下午在京闭幕。')
  
    print(result)
    ```
    ```
    [{'start': 2, 'stop': 4, 'word': '北京', 'type': 'LOC'}, {'start': 12, 'stop': 14, 'word': '苏宁', 'type': 'LOC'}, {'start': 32, 'stop': 36, 'word': '今天下午', 'type': 'T'}]
    ```
    ##### 6.MultiLayerResDYCNN(cnn4ie/dcnn)
    
    (1).训练
    ```
    from cnn4ie.dcnn.train import Train
    train = Train()
    train.train_model('config.cfg')
    ```
  ```
  Epoch: 192 | Time: 0m 4s
        Train Loss: 182.916 | Train PPL: 2.7491663642617552e+79
         Val. Loss: 463.782 |  Val. PPL: 2.618555606950152e+201
         Val. report:               precision    recall  f1-score   support

           1       1.00      1.00      1.00      5925
           2       0.99      0.98      0.98      5501
           3       0.86      0.86      0.86       174
           4       0.80      0.93      0.86        57
           5       0.84      0.79      0.81       122
           6       0.83      0.89      0.86        44
           7       0.83      0.87      0.85        62
           8       0.88      0.75      0.81        71
           9       0.92      0.71      0.80        31
          10       1.00      0.85      0.92        13

   accuracy                           0.98     12000
   macro avg       0.89      0.86      0.88     12000
   weighted avg       0.98      0.98      0.98     12000
    ```
    (2).预测
    ```
    from cnn4ie.dcnn.predict import Predict
  
    predict = Predict()
    predict.load_model_vocab('config.cfg')
    result = predict.predict('本报北京２月２８日讯记者苏宁报道：八届全国人大常委会第三十次会议今天下午在京闭幕。')
  
    print(result)
    ```
    ```
    [{'start': 2, 'stop': 4, 'word': '北京', 'type': 'LOC'}, {'start': 12, 'stop': 14, 'word': '苏宁', 'type': 'LOC'}, {'start': 32, 'stop': 36, 'word': '今天下午', 'type': 'T'}]
    ```
    ##### 7.MultiLayerStdAttnCNN(cnn4ie/stand_alone_self_attention_cnn)
    
    (1).训练
    ```
    from cnn4ie.stand_alone_self_attention_cnn.train import Train
    train = Train()
    train.train_model('config.cfg')
    ```
  ```
  Epoch: 195 | Time: 0m 3s
        Train Loss: 247.570 | Train PPL: 3.29768182789317e+107
         Val. Loss: 681.482 |  Val. PPL: 9.20623044303632e+295
         Val. report:               precision    recall  f1-score   support

           1       0.99      1.00      1.00      4539
           2       0.99      0.99      0.99      4926
           3       0.95      0.86      0.90       166
           4       0.93      0.96      0.94        52
           5       0.91      0.78      0.84       120
           6       0.93      0.97      0.95        39
           7       0.80      0.89      0.84        54
           8       0.91      0.72      0.80        68
           9       1.00      0.69      0.82        26
          10       1.00      0.90      0.95        10

   accuracy                           0.98     10000
   macro avg       0.94      0.88      0.90     10000
   weighted avg       0.98      0.98      0.98     10000
    ```
    (2).预测
    ```
    from cnn4ie.stand_alone_self_attention_cnn.predict import Predict
  
    predict = Predict()
    predict.load_model_vocab('config.cfg')
    result = predict.predict('本报北京２月２８日讯记者苏宁报道：八届全国人大常委会第三十次会议今天下午在京闭幕。')
  
    print(result)
    ```
    ```
    [{'start': 19, 'stop': 26, 'word': '全国人大常委会', 'type': 'ORG'}, {'start': 32, 'stop': 36, 'word': ' 今天下午', 'type': 'T'}, {'start': 2, 'stop': 4, 'word': '北京', 'type': 'LOC'}, {'start': 12, 'stop': 14, 'word': '苏宁', 'type': 'LOC'}]
    ```
    ##### 8.MultiLayerCSAttCNN(cnn4ie/channel_spatial_attention_cnn)     
    (1).训练
    ```
    from cnn4ie.channel_spatial_attention_cnn.train import Train
    train = Train()
    train.train_model('config.cfg')
    ```
  ```
  Epoch: 181 | Time: 0m 3s
        Train Loss: 112.922 | Train PPL: 1.1001029953413096e+49
         Val. Loss: 493.448 |  Val. PPL: 2.002428912702234e+214
         Val. report:               precision    recall  f1-score   support

           1       0.99      1.00      1.00      4539
           2       0.98      0.98      0.98      4926
           3       0.89      0.81      0.85       166
           4       0.77      0.88      0.82        52
           5       0.90      0.73      0.81       120
           6       0.84      0.92      0.88        39
           7       0.81      0.89      0.85        54
           8       0.90      0.69      0.78        68
           9       0.85      0.85      0.85        26
          10       0.82      0.90      0.86        10

   accuracy                           0.98     10000
   macro avg       0.88      0.87      0.87     10000
   weighted avg       0.98      0.98      0.98     10000
    ```
    (2).预测
    ```
    from cnn4ie.channel_spatial_attention_cnn.predict import Predict
  
    predict = Predict()
    predict.load_model_vocab('config.cfg')
    result = predict.predict('本报北京２月２８日讯记者苏宁报道：八届全国人大常委会第三十次会议今天下午在京闭幕。')
  
    print(result)
    ```
    ```
    [{'start': 2, 'stop': 4, 'word': '北京', 'type': 'LOC'}, {'start': 12, 'stop': 14, 'word': '苏宁', 'type': 'LOC'}, {'start': 32, 'stop': 36, 'word': '今天下午', 'type': 'T'}]    
    ```
    ##### 9.MultiLayerSACNN(cnn4ie/self_attention_cnn)
    (1).训练
    ```
    from cnn4ie.self_attention_cnn.train import Train
    train = Train()
    train.train_model('config.cfg')
    ```
  ```
  Epoch: 198 | Time: 0m 2s
        Train Loss: 241.123 | Train PPL: 5.227354818437855e+104
         Val. Loss: 421.708 |  Val. PPL: 1.3982772880257424e+183
         Val. report:               precision    recall  f1-score   support

           1       0.99      1.00      1.00      4539
           2       0.98      0.98      0.98      4926
           3       0.89      0.87      0.88       166
           4       0.84      0.92      0.88        52
           5       0.76      0.74      0.75       120
           6       0.88      0.95      0.91        39
           7       0.83      0.91      0.87        54
           8       0.80      0.71      0.75        68
           9       1.00      0.54      0.70        26
          10       1.00      0.70      0.82        10

   accuracy                           0.98     10000
   macro avg       0.90      0.83      0.85     10000
   weighted avg       0.98      0.98      0.98     10000
    ```
    (2).预测
    ```
    from cnn4ie.self_attention_cnn.predict import Predict
  
    predict = Predict()
    predict.load_model_vocab('config.cfg')
    result = predict.predict('本报北京２月２８日讯记者苏宁报道：八届全国人大常委会第三十次会议今天下午在京闭幕。')
  
    print(result)
    ```
    ```
    [{'start': 32, 'stop': 36, 'word': '今天下午', 'type': 'T'}, {'start': 19, 'stop': 26, 'word': '全国人大常委会', 'type': 'ORG'}, {'start': 2, 'stop': 4, 'word': '北京', 'type': 'LOC'}, {'start': 12, 'stop': 14, 'word': '苏宁', 'type': 'LOC'}]
    ```
    ##### 10.MultiLayerGroupMixedCNN(cnn4ie/mixed_depthwise_cnn)
    (1).训练
    ```
    from cnn4ie.mixed_depthwise_cnn.train import Train
    train = Train()
    train.train_model('config.cfg')
    ```
  ```
  Epoch: 200 | Time: 0m 1s
        Train Loss: 310.169 | Train PPL: 5.0653182367925945e+134
         Val. Loss: 451.143 |  Val. PPL: 8.489160946059989e+195
         Val. report:               precision    recall  f1-score   support

           1       1.00      1.00      1.00      4539
           2       0.98      0.99      0.99      4926
           3       0.93      0.83      0.88       166
           4       0.89      0.90      0.90        52
           5       0.89      0.75      0.81       120
           6       0.92      0.92      0.92        39
           7       0.91      0.93      0.92        54
           8       0.86      0.71      0.77        68
           9       1.00      0.58      0.73        26
          10       1.00      0.70      0.82        10

   accuracy                           0.99     10000
   macro avg       0.94      0.83      0.87     10000
   weighted avg       0.98      0.99      0.98     10000
    ```
    (2).预测
    ```
    from cnn4ie.mixed_depthwise_cnn.predict import Predict
  
    predict = Predict()
    predict.load_model_vocab('config.cfg')
    result = predict.predict('本报北京２月２８日讯记者苏宁报道：八届全国人大常委会第三十次会议今天下午在京闭幕。')
  
    print(result)
    ```
    ```
    [{'start': 19, 'stop': 24, 'word': '全国人大常', 'type': 'ORG'}, {'start': 2, 'stop': 4, 'word': '北京', 'type': 'LOC'}, {'start': 12, 'stop': 14, 'word': '苏宁', 'type': 'LOC'}, {'start': 32, 'stop': 36, 'word': '今天下午', 'type': 'T'}]
    ```
    ##### 11.MultiLayerMultiCNN(cnn4ie/multi_cnn)
    (1).训练
    ```
    from cnn4ie.multi_cnn.train import Train
    train = Train()
    train.train_model('config.cfg')
    ```
  ```
  Epoch: 200 | Time: 0m 1s
        Train Loss: 234.673 | Train PPL: 8.267382310706752e+101
         Val. Loss: 444.010 |  Val. PPL: 6.779999895568844e+192
         Val. report:               precision    recall  f1-score   support

           1       1.00      1.00      1.00      4539
           2       0.98      0.99      0.98      4926
           3       0.92      0.84      0.88       166
           4       0.81      0.96      0.88        52
           5       0.83      0.78      0.81       120
           6       0.86      0.95      0.90        39
           7       0.92      0.91      0.92        54
           8       0.80      0.71      0.75        68
           9       1.00      0.69      0.82        26
          10       1.00      0.70      0.82        10

   accuracy                           0.98     10000
   macro avg       0.91      0.85      0.88     10000
   weighted avg       0.98      0.98      0.98     10000
    ```
    (2).预测
    ```
    from cnn4ie.multi_cnn.predict import Predict
  
    predict = Predict()
    predict.load_model_vocab('config.cfg')
    result = predict.predict('本报北京２月２８日讯记者苏宁报道：八届全国人大常委会第三十次会议今天下午在京闭幕。')
  
    print(result)
    ```
    ```
    [{'start': 32, 'stop': 36, 'word': '今天下午', 'type': 'T'}, {'start': 20, 'stop': 25, 'word': '国人大常委', 'type': 'ORG'}, {'start': 2, 'stop': 4, 'word': '北京', 'type': 'LOC'}, {'start': 12, 'stop': 14, 'word': '苏宁', 'type': 'LOC'}]
    ```
    ##### 12.MultiLayerMixedAttCNN(cnn4ie/mixed_attention_cnn)
    (1).训练
    ```
    from cnn4ie.mixed_attention_cnn.train import Train
    train = Train()
    train.train_model('config.cfg')
    ```
  ```
  Epoch: 250 | Time: 0m 3s
	Train Loss: 107.790 | Train PPL: 6.497928061494806e+46
	 Val. Loss: 283.142 |  Val. PPL: 9.269305878179231e+122
	 Val. report:               precision    recall  f1-score   support

           1       1.00      1.00      1.00      4539
           2       0.98      0.99      0.99      4926
           3       0.91      0.84      0.87       166
           4       0.86      0.98      0.92        52
           5       0.91      0.78      0.84       120
           6       0.93      0.97      0.95        39
           7       0.88      0.91      0.89        54
           8       0.92      0.72      0.81        68
           9       1.00      0.65      0.79        26
          10       1.00      0.80      0.89        10

   accuracy                           0.99     10000
   macro avg       0.94      0.86      0.89     10000
   weighted avg       0.99      0.99      0.99     10000
    ```
    (2).预测
    ```
    from cnn4ie.mixed_attention_cnn.predict import Predict
  
    predict = Predict()
    predict.load_model_vocab('config.cfg')
    result = predict.predict('本报北京２月２８日讯记者苏宁报道：八届全国人大常委会第三十次会议今天下午在京闭幕。')
  
    print(result)
    ```
    ```
    [{'start': 32, 'stop': 36, 'word': '今天下午', 'type': 'T'}, {'start': 2, 'stop': 4, 'word': '北京', 'type': 'LOC'}, {'start': 12, 'stop': 14, 'word': '苏宁', 'type': 'LOC'}] 
    ```
    
* 
* 
* 
* 
* 
* 


## Evaluate
评估采用的是P、R、F1、PPL等。评估方法可利用scikit-learn中的precision_recall_fscore_support或classification_report。


## Install
* 安装：pip install CNN4IE
* 下载源码：
```
git clone https://github.com/jiangnanboy/CNN4IE.git
cd CNN4IE
python setup.py install
```


通过以上两种方法的任何一种完成安装都可以。如果不想安装，可以下载[github源码包](https://github.com/jiangnanboy/CNN4IE/archive/master.zip)

## Dataset

   这里利用data(来自人民日报，识别的是[ORG, PER, LOC, T, O])中的数据进行训练评估，模型1的训练及评估结果（分为带预训练向量和不带预训练向量的训练结果）见examples/mlrescnn（其它模型可自行运行评估）。
    
   预训练embedding向量：[sgns.sogou.char.bz2](https://pan.baidu.com/s/1pUqyn7mnPcUmzxT64gGpSw)

数据集的格式见[data](data/)，分为train与dev，其中source与target为中文对应的实体标注。

数据被处理成csv格式。

## Todo
持续加入更多模型......

## Cite

如果你在研究中使用了CNN4IE，请按如下格式引用：

```latex
@software{CNN4IE,
  author = {Shi Yan},
  title = {CNN4IE: Chinese Information Extraction Tool},
  year = {2021},
  url = {https://github.com/jiangnanboy/CNN4IE},
}
```

## License

CNN4IE 的授权协议为 **Apache License 2.0**，可免费用做商业用途。请在产品说明中附加CNN4IE的链接和授权协议。CNN4IE受版权法保护，侵权必究。

## Update

(1).CNN4IE 0.1.0 init commit

(2).CNN4IE 0.1.1 update self.max_len

(3).CNN4IE 0.1.2 update new model -> [MultiLayerResDSCNN]

(4).CNN4IE 0.1.3 update new model -> [MultiLayerAugmentedCNN]、[MultiLayerLambdaCNN]

(5).CNN4IE 0.1.4 update new model -> [MultiLayerResLWCNN]、[MultiLayerResDYCNN]

(6).CNN4IE 0.1.5 update new model -> [MultiLayerStdAttnCNN]

(7).CNN4IE 0.1.6 update new model -> [MultiLayerCSAttCNN]

(8).CNN4IE 0.1.7 update new model -> [MultiLayerSACNN]、[MultiLayerGroupMixedCNN]

(9).CNN4IE 0.1.8 update new model -> [MultiLayerMultiCNN]

(10).CNN4IE 0.1.9 update new model -> [MultiLayerMixedAttCNN]


## Reference

* [fairseq](https://github.com/facebookresearch/fairseq)
* [allennlp](https://github.com/allenai/allennlp)
* [Convolutional Sequence to Sequence Learning](https://arxiv.org/abs/1705.03122)
* [Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385.pdf)
* [Xception: Deep Learning with Depthwise Separable Convolutions](https://arxiv.org/pdf/1610.02357.pdf)
* [Attention Augmented Convolutional Networks](https://arxiv.org/pdf/1904.09925.pdf)
* [LambdaNetworks: Modeling long-range Interactions without Attention](https://openreview.net/forum?id=xTJEN-ggl1b)
* [Pay Less Attention with Lightweight and Dynamic Convolutions](https://arxiv.org/pdf/1901.10430.pdf)
* [Stand-Alone Self-Attention in Vision Models](https://arxiv.org/pdf/1906.05909.pdf)
* [CBAM: Convolutional Block Attention Module](https://arxiv.org/pdf/1807.06521.pdf)
* [Self-Attention Generative Adversarial Networks](https://arxiv.org/pdf/1805.08318.pdf)
* [MixConv: Mixed Depthwise Convolutional Kernels](https://arxiv.org/pdf/1907.09595.pdf)
* [Character-Level Translation with Self-attention](https://arxiv.org/pdf/2004.14788.pdf)
* [ConvBERT: Improving BERT with Span-based Dynamic Convolution](https://arxiv.org/pdf/2008.02496.pdf)
* https://github.com/leaderj1001/LambdaNetworks
* https://github.com/leaderj1001/Attention-Augmented-Conv2d
* https://github.com/pytorch/fairseq
* https://github.com/leaderj1001/Stand-Alone-Self-Attention
* https://github.com/luuuyi/CBAM.PyTorch
* https://github.com/Jongchan/attention-module
* https://github.com/fastai/fastai2/blob/master/fastai2/layers.py
* https://github.com/leaderj1001/Mixed-Depthwise-Convolutional-Kernels
* https://github.com/CharizardAcademy/convtransformer
* https://github.com/huggingface/transformers/tree/master/src/transformers/models/convbert


