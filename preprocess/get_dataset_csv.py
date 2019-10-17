
# coding: utf-8

# In[27]:


import os
import json
import numpy as np
import pandas as pd


# ## Definition of some constants
# | Item          | Data Type             | Description |
# | :-----------: | :-------------------: | :---------: |
# | data_root     | String                | The root directory to save images |
# | split_radio   | Tuple of 3 elements   | The ratio of the total dataset to use for training, validation and test respectively |
# | min_num_images_per_class | Integer | Classes with fewer images will be removed from the whole dataset |
# | min_num_train_images_per_class | Integer | Minimal number of images per class in the training set, to guarantee training performance. **Ensure that `min_num_train_images_per_class` $\le$ `min_num_images_per_class`** |
# | random_seed | Integer | The random seed | 

# In[28]:


data_root = os.path.expanduser('~/vggface3d_sm')
assert os.path.exists(data_root),     'Dataset directory not found: %s' % data_root

split_radio = (0.8, 0.1, 0.1)
assert sum(split_radio) == 1
min_num_images_per_class = 10
min_num_train_images_per_class = 5

random_seed = 2333
np.random.seed(random_seed)


# ## Generating CSV Files
# CSVs are saved in the `data_root` directory, i.e., `data_root/train.csv`, `data_root/eval.csv`, `data_root/test.csv`.
# ### Overview of DataRoot
# ```text
# vggface3d_sm
# |── train.csv
# |── eval.csv
# |── test.csv
# |── dirty.csv
# ├── n000853
# │   ├── 0001_03.npy
# │   ├── 0001_03.png
# │   ├── 0002_01.npy
# │   ├── 0002_01.png
# │   ├── 0003_01.npy
# │   ├── 0003_01.png
# │   ├── 0004_01.npy
# │   ├── ......
# ```

# In[29]:


def _get_data_of_one_class(_cls_name, shuffle=True):
    _cls_dir = os.path.realpath(os.path.join(data_root, _cls_name))
    assert os.path.exists(_cls_dir) and os.path.isdir(_cls_dir)
    _data = []
    for file in os.listdir(_cls_dir):
        file_name, file_ext = os.path.splitext(file)
        if file_ext != '.png':
            continue
        if not os.path.exists(os.path.join(_cls_dir, '%s.npy' % file_name)):
            print("%s exists but %s can not be found" % (
                os.path.join(_cls_dir, '%s.png' % file_name), 
                os.path.join(_cls_dir, '%s.npy' % file_name)))
            continue
        _data.append([os.path.join(_cls_dir, '%s.png' % file_name), 
                      os.path.join(_cls_dir, '%s.npy' % file_name),
                      _cls_name])
    if shuffle:
        np.random.shuffle(_data)
    return _data


# train_data, eval_data, test_data, dirty_data = [[]] * 4
train_data, eval_data, test_data, dirty_data = [], [], [], []

for cls_name in os.listdir(data_root):
    # Skip if it is not a folder
    if not os.path.isdir(os.path.join(data_root, cls_name)):
        continue
    cls_data = _get_data_of_one_class(cls_name, shuffle=True)
    if len(cls_data) < min_num_images_per_class:
        dirty_data.extend(cls_data)
        continue
    
    num_train_images = max(min_num_train_images_per_class, 
                           int(len(cls_data) * split_radio[0]))
    num_eval_images = int((len(cls_data) - num_train_images) * 
                          split_radio[1]/(1-split_radio[0]))
    num_test_images = len(cls_data) - num_train_images - num_eval_images
    
    train_data.extend(cls_data[:num_train_images])
    eval_data.extend(cls_data[num_train_images:num_train_images+num_eval_images])
    test_data.extend(cls_data[-num_test_images:])

train_data = np.array(train_data)
train_df = pd.DataFrame({'rgb_image_path' : train_data[:, 0] if len(train_data)>0 else [], 
                         'dep_image_path' : train_data[:, 1] if len(train_data)>0 else [], 
                         'cls_name': train_data[:, 2] if len(train_data)>0 else []})
train_df.to_csv(os.path.join(data_root, 'train.csv'), index=False, sep=',')

eval_data = np.array(eval_data)
eval_df = pd.DataFrame({'rgb_image_path' : eval_data[:, 0] if len(eval_data)>0 else [], 
                        'dep_image_path' : eval_data[:, 1] if len(eval_data)>0 else [], 
                        'cls_name': eval_data[:, 2] if len(eval_data)>0 else []})
eval_df.to_csv(os.path.join(data_root, 'eval.csv'), index=False, sep=',')

test_data = np.array(test_data)
test_df = pd.DataFrame({'rgb_image_path' : test_data[:, 0] if len(test_data)>0 else [], 
                        'dep_image_path' : test_data[:, 1] if len(test_data)>0 else [], 
                        'cls_name': test_data[:, 2] if len(test_data)>0 else []})
test_df.to_csv(os.path.join(data_root, 'test.csv'), index=False, sep=',')

dirty_data = np.array(dirty_data)
dirty_df = pd.DataFrame({'rgb_image_path' : dirty_data[:, 0] if len(dirty_data)>0 else [], 
                         'dep_image_path' : dirty_data[:, 1] if len(dirty_data)>0 else [], 
                         'cls_name': dirty_data[:, 2] if len(dirty_data)>0 else []})
dirty_df.to_csv(os.path.join(data_root, 'dirty.csv'), index=False, sep=',')
print("everything finished!")


# In[30]:


train_df.head()


# In[30]:




