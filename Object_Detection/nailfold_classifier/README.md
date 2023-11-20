# Pytorch Image Classifier Template
This code modifies the output layer of image classification network commonly used in pytorch. The modified model can be used to process any number of image classification data. At the same time, the pre-training parameters and re-training parameters are distinguished for fine-tuning and component training of parameters. We have modified the following network structure:
<table border="1">
    <tr>
        <td>MobileNet</td>
        <td>mobilenet_v2</td>
        <td>mobilenet_v3_small</td>
        <td>mobilenet_v3_large</td>
    </tr>
    <tr>
        <td>ResNet</td>
        <td>resnet18</td>
        <td>resnet34</td>
        <td>resnet50</td>
        <td>resnet101</td>
    </tr>
    <tr>
		<td>ResNeXt</td>
        <td>resnext50_32x4d</td>
        <td>resnext101_32x8d</td>
    </tr>
    <tr>
		<td>DenseNet</td>
        <td>densenet121</td>
        <td>densenet161</td>
        <td>densenet169</td>
    </tr>
    <tr>
		<td>ShuffleNet</td>
        <td>shufflenet_v2_x0_5</td>
        <td>shufflenet_v2_x1_0</td>
    </tr>
    <tr>
		<td>SqueezeNet</td>
        <td>squeezenet1_0</td>
        <td>squeezenet1_1</td>
    </tr>
    <tr>
		<td>WideResNet</td>
        <td>wide_resnet50_2</td>
        <td>wide_resnet101_2</td>
    </tr>
</table>

# Blog
* [here](https://blog.csdn.net/cxx654/article/details/118254718?spm=1001.2014.3001.5501)

# Requirement
* torch==1.9.0
* torchvision==0.10.0

# Usage
* modify data/classifier.yaml
    - classes: class list
    - dataset_dir: dataset path
      - Cat
        - 1.img
        - 2.img
      - Dog
        - 1.img
        - 2.img
    - save_dir: model save path
    - train_size: random choice sample number per class
* model train: `python train_classifier_model.py`
* model inference: `python model_inference.py`

# Finetune Cifar100 Results
we use the ./data/ciassifier_cifar100.yaml and train_cifar100_model.py to train cifar100 dataset. training curve in ./images dir.
<table border="1">
    <tr>
        <td>model</td>
        <td>accuracy</td>
        <td>fps(batch=1, img_size=224x224)</td>
        <td>GPU memory</td>
        <td>GPU Peak(Tesla P40)</td>
        <td>model_size(bit size)</td>
        <td>Total Params</td>
    </tr>
    <tr>
        <td>mobilenet_v2</td>
        <td>0.7123</td>
        <td>164</td>
        <td>847M</td>
        <td>43%</td>
        <td>9.3M</td>
        <td>2,236,682</td>
    </tr>
    <tr>
        <td>mobilenet_v3_small</td>
        <td>0.7112</td>
        <td>144</td>
        <td>847M</td>
        <td>36%</td>
        <td>6.4M</td>
        <td>1,528,106</td>
    <tr>
        <td>mobilenet_v3_large</td>
        <td>0.7521</td>
        <td>138</td>
        <td>855M</td>
        <td>50%</td>
        <td>17M</td>
        <td>4,214,842</td>
    </tr>
    <tr>
        <td>resnet18</td>
        <td>0.7139</td>
        <td>236</td>
        <td>913M</td>
        <td>56%</td>
        <td>43M</td>
        <td>11,181,642</td>
    </tr>
    <tr>
        <td>resnet34</td>
        <td>0.7573</td>
        <td>182</td>
        <td>955M</td>
        <td>59%</td>
        <td>82M</td>
        <td>21,289,802</td>
    </tr>
    <tr>
        <td>resnet50</td>
        <td>0.7683</td>
        <td>123</td>
        <td>967M</td>
        <td>74%</td>
        <td>91M</td>
        <td>23,528,522</td>
    </tr>
    <tr>
        <td>resnet101</td>
        <td>0.8025</td>
        <td>81</td>
        <td>1041M</td>
        <td>89%</td>
        <td>164M</td>
        <td>42,520,650</td>
    </tr>
    <tr>
        <td>resnext50_32x4d</td>
        <td>0.7806</td>
        <td>67</td>
        <td>933M</td>
        <td>73%</td>
        <td>89M</td>
        <td>23,000,394</td>
    </tr>
    <tr>
        <td>resnext101_32x8d</td>
        <td>0.8198</td>
        <td>31</td>
        <td>1195M</td>
        <td>88%</td>
        <td>333M</td>
        <td>86,762,826</td>
    </tr>
    <tr>
        <td>densenet121</td>
        <td>0.768</td>
        <td>69</td>
        <td>869M</td>
        <td>66%</td>
        <td>28M</td>
        <td>6,964,106</td>
    </tr>
    <tr>
        <td>densenet161</td>
        <td>0.8027</td>
        <td>45</td>
        <td>957M</td>
        <td>86%</td>
        <td>104M</td>
        <td>26,494,090</td>
    </tr>
    <tr>
        <td>densenet169</td>
        <td>0.7868</td>
        <td>50</td>
        <td>889M</td>
        <td>65%</td>
        <td>50M</td>
        <td>12,501,130</td>
    </tr>
    <tr>
        <td>shufflenet_v2_x0_5</td>
        <td>0.2203</td>
        <td>136</td>
        <td>843M</td>
        <td>24%</td>
        <td>1.9M</td>
        <td>352,042</td>
    </tr>
    <tr>
        <td>shufflenet_v2_x1_0</td>
        <td>0.3329</td>
        <td>150</td>
        <td>845M</td>
        <td>32%</td>
        <td>5.4M</td>
        <td>1,263,854</td>
    </tr>
    <tr>
        <td>squeezenet1_0</td>
        <td>0.5605</td>
        <td>292</td>
        <td>837M</td>
        <td>47%</td>
        <td>3.1M</td>
        <td>740,554</td>
    </tr>
    <tr>
        <td>squeezenet1_1</td>
        <td>0.575</td>
        <td>266</td>
        <td>837M</td>
        <td>41%</td>
        <td>3.0M</td>
        <td>727,626</td>
    </tr>
    <tr>
        <td>wide_resnet50_2</td>
        <td>0.789</td>
        <td>90</td>
        <td>1227M</td>
        <td>88%</td>
        <td>257M</td>
        <td>66,854,730</td>
    </tr>
    <tr>
        <td>wide_resnet101_2</td>
        <td>0.8122</td>
        <td>50</td>
        <td>1469M</td>
        <td>93%</td>
        <td>478M</td>
        <td>124,858,186</td>
    </tr>
</table>