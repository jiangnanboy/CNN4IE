# CNN4IE

中文信息抽取工具。使用CNN的不同变体进行信息抽取，未来会持续加入不同模型。该项目使用pytorch，python开发。

**CNN4IE**根据CNN的各种改进版本，对不同模型块进行融合，并将其用于中文信息抽取中。

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
* MultiLayerResCNN(src/mlrescnn)：多层残差CNN(+CRF),此模型利用ConvSeq2Seq中的encoder部分进行多层叠加，后面可接CRF。
#### Usage
- 相关参数的配置

    [config.cfg](cnn4ie/mlrescnn/config.cfg)，在训练或预测时加载此文件的位置。

- 训练(支持加载预训练的embedding向量)
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
      
- 预测

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

   这里利用data(来自人民日报，识别的是[ORG, PER, LOC, T, O])中的数据进行训练评估，训练及评估结果见examples/mlrescnn，分为带预训练向量和不带预训练向量的训练结果。
    
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

## Reference

* [fairseq](https://github.com/facebookresearch/fairseq)
* [allennlp](https://github.com/allenai/allennlp)
* [Convolutional Sequence to Sequence Learning](https://arxiv.org/abs/1705.03122)
* [Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385.pdf)