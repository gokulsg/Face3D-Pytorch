# Face3D: 3D Face Recognition 


This is a 3D Face Recognizer, a.k.a. 2.5D Face Recognizer in many cases, implemented with `PyTorch`. 

Another implementation with tensorflow can be found [there](https://github.com/xingwxiong/Face3D).

Given a pair of an RGB image and a depth image, that is, a four-dimensional image, the recognizer needs to recognize the face in the image.

The following is a detailed description of the dataset.

## TODO
- [ ] Multi-GPU training
- [ ] Distributed training
- [ ] Evaluation and predition script
- [ ] The links of the pretrained models

## Dataset 
Dataset is available from [here](http://125.39.136.212:8484/3dvggface2_1.tar.gz).

The data set contains **403,067** pairs of face images of **1208** people. Each pair of face images is registered and contains an RGB image and a depth image.

<div style="text-align:center;" align="center">
    <!-- Image Caption Template -->
    <div style="padding:0; margin-bottom: 0;">
        <img style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08); margin: 0 10px;"
        src="./README/rgb_0001_03.jpg" alt="the RGB image">
         <img style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08); margin: 0 10px;"
        src="./README/dep_0001_03.jpg" alt="the depth image">
    </div>
    <div style="border-bottom: 1px solid #d9d9d9; display: inline-block; padding: 0; color: #999"><strong> The RGB image and the corresponding depth image.</strong></div>
</div>

## Model
I use `ResNet50` models (CNN) for feature extraction, whose input channel is slighted modified so that 4D channel images can be used as data inputs.

I am about to compare the performance of some different classification loss functions, such as `softmax`, `triplet loss`, etc.

Below are some of the trained models and their accuracy on the dataset. **The list will continue to be updated**, please keep your attention.

| Model Name | Accuracy | Architecture | Description |
| :--------: | :------: | :----------: | :---------: |
| RGB-D-ResNet50-from-scratch.pkl | 85.11% | RGB-D ResNet50 | <details><summary><i>Training from scratch.</i></summary><ul><li>Take RGB-D images as input, , i.e, 4-channel input.</li></ul></details> |
| RGB-D-ResNet50-from-imagenet.pkl | 93.68% | RGB-D ResNet50 | <details><summary><i>Pretrained on imagenet.</i></summary><ul><li>Take RGB-D images as input, i.e, 4-channel input. </li><li>Pretrain on imagenet, then fine tune on the RGB-D dataset.</li></ul></details> |


## Preprocess
### Face alignment
The model should be fed with images of fixed size, therefore we need to perform face alignment first.

Please refer to the code of [davidsandberg/facenet](https://github.com/davidsandberg/facenet).

```bash
# align.sh

export PYTHONPATH=${PWD}/src

python preprocess/align/align_dataset_mtcnn.py \
    --input_dir /mnt/sdb/vggface3 \
    --output_dir /mnt/sdb/vggface3_align \
    --image_size 182 \
    --margin 44 \
    --random_order \
    --thread_num 3 \
    --gpu_memory_fraction 0.88
```

### Dataset splitting
To split the whole dataset randomly into 3 sub-datasets, (i.e., training dataset, evaluation dataset, test dataset), by generating 3 corresponding csv files to record the paths and labels of each images.

```bash
python preprocess/get_dataset_csv.py
```

After that, the file structure of the data set is as follows. 

```text
vggface3d
|── train.csv
|── eval.csv
|── test.csv
|── dirty.csv
├── n000853
│   ├── 0001_03.npy
│   ├── 0001_03.png
│   ├── 0002_01.npy
│   ├── 0002_01.png
│   ├── 0003_01.npy
│   ├── 0003_01.png
│   ├── 0004_01.npy
│   ├── ......
```

## Train
### Train with softmax
Multi-GPU training will be supported soon.

```bash
python train_softmax.py
```
**Note**: Currently, all the parameters are hard-coding, and the argument parser will be implemented soon.

## Evaluation
Available soon.

## Predict
Available soon.

