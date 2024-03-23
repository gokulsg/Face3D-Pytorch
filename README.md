# Face3D: 3D Face Recognition

This is a 3D Face Recognizer, a.k.a. 2.5D Face Recognizer in many cases, implemented with `PyTorch`. 

Another implementation with tensorflow can be found [there](https://github.com/xingwxiong/Face3D).

Given a pair of an RGB image and a depth image, that is, a four-dimensional image, the recognizer needs to recognize the face in the image.

The following is a detailed description of the dataset.

## TODO

- [ ] Pretrained RGB models (3-channel input)
- [x] Pretrained RGB-D models (4-channel input)
- [x] Data-parallel Multi-GPU training 
- [ ] Data-parallel distributed training (DDL)
- [ ] Prediction script
- [ ] Training with triplet loss
- [ ] Face encoder to generate face embedding

## Dataset
Dataset is available from [here]
The data set contains **403,067** pairs of face images of **1,208** people. Each pair of face images is registered and contains an RGB image and a depth image.

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

| Model Name | Architecture | Accuracy | Descriptions |
| :--------: | :----------: | :------: | :----------- |
| RGB-ResNet50-from-imagenet.pkl | RGB ResNet50 | 94.47% | <details><summary><i>Pretrained on ImageNet.</i></summary><ul><li>Only take RGB images as input, without considering depth images.</li><li>Pretrain on imagenet, then fine tune on the RGB dataset.</li></ul></details> |
| [RGB-D-ResNet50-from-scratch.pkl](https://drive.google.com/open?id=1qwbTikrF04mJ4Z170aWefHvpP3yfqJim) | RGB-D ResNet50 | 88.36% | <details><summary><i>Training from scratch.</i></summary><ul><li>Take RGB-D images as input, , i.e, 4-channel input.</li><li>Pretrain on imagenet, then fine tune on the RGB-D dataset.</li><li style="width: 300px;overflow: hidden;white-space: nowrap; text-overflow: ellipsis;" title="1694c02b18ba3c55f55593eb385715291e3e0fd7"><strong>SHA1:</strong> 1694c02b18ba3c55f55593eb385715291e3e0fd7</li></ul></details> |
| [RGB-D-ResNet50-from-imagenet.pkl](https://drive.google.com/open?id=1CIPwX0l5Q5IB_CaitCO-Hvlf67A1c6eg) | RGB-D ResNet50 | 94.64% | <details><summary><i>Pretrained on ImageNet.</i></summary><ul><li>Take RGB-D images as input, i.e, 4-channel input. </li><li style="width: 300px;overflow: hidden;white-space: nowrap; text-overflow: ellipsis;" title="042c9f8e444975b0915d16dccdcb87b55b36e4cb"><strong>SHA1:</strong> 042c9f8e444975b0915d16dccdcb87b55b36e4cb</li></ul></details> |

## Install

```bash
pip install -r requirements
```

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
python train_softmax.py --train_dataset_csv '~/vggface3d_sm/train.csv' \
    --eval_dataset_csv '~/vggface3d_sm/eval.csv' \
    --pretrained_on_imagenet \
    --input_channels 4 \
    --num_of_classes 1200 \
    --num_of_epochs 50 \
    --num_of_workers 8 \
    --log_base_dir './logs'
```

<details><summary>Click to see the usage.</summary>
<pre lang="text">
usage: train_softmax.py [-h] [--train_dataset_csv TRAIN_DATASET_CSV]
                        [--eval_dataset_csv EVAL_DATASET_CSV]
                        [--pretrained_on_imagenet]
                        [--pretrained_model_path PRETRAINED_MODEL_PATH]
                        [--pretrained_optim_path PRETRAINED_OPTIM_PATH]
                        [--input_channels INPUT_CHANNELS]
                        [--num_of_classes NUM_OF_CLASSES]
                        [--num_of_epochs NUM_OF_EPOCHS]
                        [--image_size IMAGE_SIZE] [--batch_size BATCH_SIZE]
                        [--num_of_workers NUM_OF_WORKERS]
                        [--logs_base_dir LOGS_BASE_DIR]
optional arguments:
  -h, --help            show this help message and exit
  --train_dataset_csv TRAIN_DATASET_CSV
                        The path of csv file where to write paths of training
                        images.
  --eval_dataset_csv EVAL_DATASET_CSV
                        The path of csv file where to write paths of
                        validation images.
  --pretrained_on_imagenet
                        (bool) Whether to load the imagenet pretrained model.
  --pretrained_model_path PRETRAINED_MODEL_PATH
                        Load a pretrained model before training starts.
  --pretrained_optim_path PRETRAINED_OPTIM_PATH
                        Load a optimizer before training starts.
  --input_channels INPUT_CHANNELS
                        Number of channels of the first input layer.
  --num_of_classes NUM_OF_CLASSES
                        Number of channels of the last output layer.
  --num_of_epochs NUM_OF_EPOCHS
                        Number of epochs to run.
  --image_size IMAGE_SIZE
                        Image size (height, width) in pixels.
  --batch_size BATCH_SIZE
                        Number of images to process in a batch.
  --num_of_workers NUM_OF_WORKERS
                        Number of subprocesses to use for data loading.
  --logs_base_dir LOGS_BASE_DIR
                        Directory where to write event logs and save models.
</pre>
</details>

## Evaluation

```bash
python evaluation.py \
    --pretrained_model_path ./RGB-D-ResNet50-from-scratch.pkl \
    --num_of_workers 8
```

<details><summary>Click to see the usage.</summary>
<pre lang="text">
usage: evaluation.py [-h] [--test_dataset_csv TEST_DATASET_CSV]
                     [--pretrained_model_path PRETRAINED_MODEL_PATH]
                     [--input_channels INPUT_CHANNELS]
                     [--num_of_classes NUM_OF_CLASSES]
                     [--image_size IMAGE_SIZE] [--batch_size BATCH_SIZE]
                     [--num_of_workers NUM_OF_WORKERS]
optional arguments:
  -h, --help            show this help message and exit
  --test_dataset_csv TEST_DATASET_CSV
                        The path of csv file where to write paths of test
                        images.
  --pretrained_model_path PRETRAINED_MODEL_PATH
                        The path of the pretrained model.
  --input_channels INPUT_CHANNELS
                        Number of channels of the first input layer.
  --num_of_classes NUM_OF_CLASSES
                        Number of channels of the last output layer.
  --image_size IMAGE_SIZE
                        Image size (height, width) in pixels.
  --batch_size BATCH_SIZE
                        Number of images to process in a batch.
  --num_of_workers NUM_OF_WORKERS
                        Number of subprocesses to use for data loading.
</pre></details>

## Predict
Available soon.

