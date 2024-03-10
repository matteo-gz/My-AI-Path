---
title: 辨别鸟类模型
type: docs
---

## 实践来源

https://www.kaggle.com/code/jhoward/is-it-a-bird-creating-a-model-from-your-own-data

在过去识别一个照片中是否有鸟类,需要一个研究团和5年的时间(一个梗图).

## 前置工作

fastai库
`pip install -Uqq fastai`

确保更新,减少兼容性问题

## 名词解释

{{< expand "Fine-tune">}}
```
Fine-tune 是一个英文单词，意思是 微调。在机器学习领域，fine-tune 指的是对已经训练好的模型进行进一步调整，以使其在新的任务上表现更好。
```

{{< /expand >}}


{{< expand "Pretrained" >}}

```
Pretrained 是一个英文单词，意思是 预训练。在机器学习领域，pretrained 指的是使用一个模型在大量数据上进行训练，然后将训练好的模型参数作为另一个模型的初始化参数。
```
{{< /expand >}}



## 关于data block
数据块与模型的关联上(数据如何进入我们的模型),在此处的探讨上.
我们似乎跳过了 neural network(神经网络), matrix multiplication(矩阵相乘),gradients(梯度).

但是在实践过程中深度学习社区已经找到了相当小量的模型类型,几乎满足我们的构建需求.

fast.ai库为我们构建正确类型的模型.

### 数据块设计
在数百个项目中,为了使数据达到正确的形状,每个项目进行了哪些更改.

### 概念
- training set 训练集(用于创建模型的图像)
- validation set 验证集(在训练期间不使用)

###  数据集定义
```py
dls = DataBlock(
    blocks=(ImageBlock, CategoryBlock),
    get_items=get_image_files, 
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=parent_label,
    item_tfms=[Resize(192, method='squish')]
).dataloaders(path, bs=32)
```

- 数据块
  - 我们有什么样的输入? -- 图像
  - 有什么样的输出? -- 类别 (bird or forest)
- 那么这个模型中有哪些物品呢?  -- 图像文件列表
- 留出一些数据来测试模型的准确性(验证集) -- 随机20%
- 我们如何知道一张照片的正确标签,我们如何知道这是一张鸟类照片还是一张森林照片? -- 标签的来源,通过图片返回其文件夹(包含标签)
- 项目转换;图片大小转换为192 by 192像素. 通过 squish

以上步骤通过GPU 并行执行出一个 "batch" or "mini batch"

分离器的类型大全
- https://docs.fast.ai/tutorial.datablock.html
- https://docs.fast.ai/data.block.html

图像模型
- https://github.com/huggingface/pytorch-image-models

## 代码
### 前置依赖
```shell
pip install -Uqq fastai duckduckgo_search
```
### main.py
{{< expand "code">}}
```py
import multiprocessing
if __name__ == '__main__':

    # add this import

    multiprocessing.freeze_support()

    from duckduckgo_search import DDGS
    from fastcore.all import *

    ddgs = DDGS()
    def search_images(term, max_images=30):
        print(f"Searching for '{term}'")
    #     return L(ddg_images(term, max_results=max_images)).itemgot('image')
        return L(ddgs.images(keywords=term, max_results=max_images)).itemgot('image')


    urls = search_images('bird photos', max_images=1)
    urls[0]
    print(urls[0])

    from fastdownload import download_url
    dest = 'bird.jpg'
    download_url(urls[0], dest, show_progress=False)

    from fastai.vision.all import *
    im = Image.open(dest)
    im.to_thumb(256,256)
    download_url(search_images('forest photos', max_images=1)[0], 'forest.jpg', show_progress=False)
    Image.open('forest.jpg').to_thumb(256,256)

    searches = 'forest','bird'
    path = Path('bird_or_not')
    from time import sleep

    from fastai.vision.all import *


    for o in searches:
        # continue
        dest = (path/o)
        dest.mkdir(exist_ok=True, parents=True)
        download_images(dest, urls=search_images(f'{o} photo'))
        sleep(10)  # Pause between searches to avoid over-loading server
        download_images(dest, urls=search_images(f'{o} sun photo'))
        sleep(10)
        download_images(dest, urls=search_images(f'{o} shade photo'))
        sleep(10)
        resize_images(path/o, max_size=400, dest=path/o)


    failed = verify_images(get_image_files(path))
    failed.map(Path.unlink)
    len(failed)


    dls = DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        get_items=get_image_files,
        splitter=RandomSplitter(valid_pct=0.2, seed=42),
        get_y=parent_label,
        item_tfms=[Resize(192, method='squish')]
    ).dataloaders(path, bs=32)

    dls.show_batch(max_n=6)

    learn = vision_learner(dls, resnet18, metrics=error_rate)
    learn.fine_tune(3)


    is_bird,_,probs = learn.predict(PILImage.create('bird.jpg'))
    print(f"This is a: {is_bird}.")
    print(f"Probability it's a bird: {probs[0]:.4f}")

    is_bird, _, probs = learn.predict(PILImage.create('forest.jpg'))
    print(f"This is a: {is_bird}.")
    print(f"Probability it's a bird: {probs[0]:.4f}")
```
{{< /expand >}}
### 文件目录
```
.
├── bird.jpg
├── bird_or_not
│   ├── bird
│   └── forest
├── forest.jpg
├── main.py
└── venv
    ├── bin
    ├── lib
    └── pyvenv.cfg
```
### 运行脚本
```shell
/usr/bin/python3 main.py
```
### 输出结果
```shell
epoch     train_loss  valid_loss  error_rate  time    
0         0.732300    0.141044    0.046512    00:12                                               
epoch     train_loss  valid_loss  error_rate  time    
0         0.183334    0.021051    0.000000    00:09                                               
1         0.115255    0.014208    0.000000    00:04                                               
2         0.075642    0.008127    0.000000    00:03                                               
This is a: bird.                                                                
Probability it's a bird: 1.0000
This is a: forest.                                                              
Probability it's a bird: 0.0000
```