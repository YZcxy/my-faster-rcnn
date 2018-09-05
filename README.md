# my-faster-rcnn  

这是一份**注释**比较充足的faster-rcnn代码，仅用于学习交流  
代码**完全**参考于[tf-faster-rcnn](https://github.com/endernewton/tf-faster-rcnn) 和 [Faster-RCNN-TensorFlow-Python3.5](https://github.com/dBeker/Faster-RCNN-TensorFlow-Python3.5)

## 怎 么 使 用  
 
`lib/` 下几乎涵盖了所有的代码内容   
`tools/` 下包含train和demo两个方法，用于训练和测试   
`data/` 下应该包含VGG16的参数，VOCDevkit2007训练集，测试数据demo  

**所以，如果你想运行代码**  

1- 各种环境配置，自行捣鼓，python建议使用3.5

2- 在这个路径下取下载VOCDevkit2007训练集 [Link]( https://github.com/rbgirshick/py-faster-rcnn#beyond-the-demo-installation-for-training-and-testing-models)  
然后放在这两个目录   
`data/VOCDevkit2007/annotations_cache`
`data/VOCDevkit2007/VOC2007`

3- 下载VGG16已训练好的参数[here](http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz)   
然后放在这个目录 `data\imagenet_weights\vgg16.ckpt`


## 完

其实运行起来并没有什么用，想熟悉网络还得一行一行读代码才是，炼丹师们，加油
