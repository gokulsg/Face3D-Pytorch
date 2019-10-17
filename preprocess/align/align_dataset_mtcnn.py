"""Performs face alignment and stores face thumbnails in the output directory."""
# MIT License
# 
# Copyright (c) 2016 David Sandberg
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy import misc
import sys
import os
import argparse
import tensorflow as tf
import numpy as np
import facenet
import align.detect_face
import random
from time import sleep
import threading
from tqdm import tqdm


class AlignThread(threading.Thread):
    num_aligned_img = 0
    num_total_img = 0
    threadLock = threading.Lock()
    threads = []
    output_dir = './vggface3d_align'
    random_order = True
    detect_multiple_faces = False
    margin = 0
    image_size = 182
    dataset = None
    pnet, onet, rnet = [None]*3
    pbar = None

    @classmethod
    def initialization(cls, args):
        cls.output_dir = os.path.expanduser(args.output_dir)
        cls.random_order = args.random_order
        cls.margin = args.margin
        cls.image_size = args.image_size
        cls.detect_multiple_faces = args.detect_multiple_faces
        if not os.path.exists(cls.output_dir):
            os.makedirs(cls.output_dir)
        # Store some git revision info in a text file in the log directory
        src_path, _ = os.path.split(os.path.realpath(__file__))
        facenet.store_revision_info(src_path, cls.output_dir, ' '.join(sys.argv))
        cls.dataset = facenet.get_dataset(args.input_dir)
        cls.num_total_img = 0
        for cls_name in cls.dataset:
            cls.num_total_img += len(cls_name.image_paths)
        cls.pbar = tqdm(total=cls.num_total_img)
        print('Creating networks and loading parameters')
        with tf.Graph().as_default():
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory_fraction)
            sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
            with sess.as_default():
                cls.pnet, cls.rnet, cls.onet = align.detect_face.create_mtcnn(sess, None)

    def __init__(self, threadID):
        threading.Thread.__init__(self)
        self.threadID = threadID

    def run(self):
        # sleep(random.random())
        minsize = 20  # minimum size of face
        threshold = [0.6, 0.7, 0.7]  # three steps's threshold
        factor = 0.709  # scale factor

        # Add a random key to the filename to allow alignment using multiple processes
        random_key = np.random.randint(0, high=99999)
        bounding_boxes_filename = os.path.join(AlignThread.output_dir, 'bounding_boxes_%05d.txt' % random_key)

        with open(bounding_boxes_filename, "w") as text_file:
            nrof_successfully_aligned = 0
            if AlignThread.random_order:
                random.shuffle(AlignThread.dataset)
            for cls_name in AlignThread.dataset:
                output_class_dir = os.path.join(AlignThread.output_dir, cls_name.name)
                if not os.path.exists(output_class_dir):
                    os.makedirs(output_class_dir)
                    if AlignThread.random_order:
                        random.shuffle(cls_name.image_paths)
                for image_path in cls_name.image_paths:
                    depth_path = os.path.splitext(image_path)[0] + ".npy"
                    filename = os.path.splitext(os.path.split(image_path)[1])[0]
                    output_filename = os.path.join(output_class_dir, filename + '.png')
                    if not os.path.exists(output_filename):
                        try:
                            img = misc.imread(image_path)
                            dep = np.load(depth_path)  # Shape: (H, W）
                            # dep = np.expand_dims(dep, axis=2)   # Shape: (H, W, 1）
                        except (IOError, ValueError, IndexError) as e:
                            error_message = '{}: {}'.format(image_path, e)
                            print(error_message)
                        else:
                            if img.ndim < 2:
                                print('Unable to align "%s"' % image_path)
                                text_file.write('%s\n' % output_filename)
                                continue
                            if img.ndim == 2:
                                img = facenet.to_rgb(img)
                            img = img[:, :, 0:3]

                            bounding_boxes, _ = align.detect_face.detect_face(img, minsize,
                                                                              AlignThread.pnet,
                                                                              AlignThread.rnet,
                                                                              AlignThread.onet,
                                                                              threshold, factor)
                            nrof_faces = bounding_boxes.shape[0]
                            if nrof_faces > 0:
                                det = bounding_boxes[:, 0:4]
                                det_arr = []
                                img_size = np.asarray(img.shape)[0:2]
                                if nrof_faces > 1:
                                    if AlignThread.detect_multiple_faces:
                                        for i in range(nrof_faces):
                                            det_arr.append(np.squeeze(det[i]))
                                    else:
                                        bounding_box_size = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
                                        img_center = img_size / 2
                                        offsets = np.vstack([(det[:, 0] + det[:, 2]) / 2 - img_center[1],
                                                             (det[:, 1] + det[:, 3]) / 2 - img_center[0]])
                                        offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
                                        # some extra weight on the centering
                                        index = np.argmax(bounding_box_size - offset_dist_squared * 2.0)
                                        det_arr.append(det[index, :])
                                else:
                                    det_arr.append(np.squeeze(det))

                                for i, det in enumerate(det_arr):
                                    det = np.squeeze(det)
                                    bb = np.zeros(4, dtype=np.int32)
                                    bb[0] = np.maximum(det[0] - AlignThread.margin / 2, 0)
                                    bb[1] = np.maximum(det[1] - AlignThread.margin / 2, 0)
                                    bb[2] = np.minimum(det[2] + AlignThread.margin / 2, img_size[1])
                                    bb[3] = np.minimum(det[3] + AlignThread.margin / 2, img_size[0])
                                    cropped_img = img[bb[1]:bb[3], bb[0]:bb[2], :]
                                    cropped_dep = dep[bb[1]:bb[3], bb[0]:bb[2]]
                                    scaled_img = misc.imresize(cropped_img, (AlignThread.image_size, AlignThread.image_size),
                                                               interp='bilinear')
                                    scaled_dep = misc.imresize(cropped_dep, (AlignThread.image_size, AlignThread.image_size),
                                                               interp='bilinear')

                                    nrof_successfully_aligned += 1
                                    AlignThread.threadLock.acquire()
                                    AlignThread.num_aligned_img += 1
                                    AlignThread.pbar.update(1)
                                    AlignThread.threadLock.release()

                                    filename_base, file_extension = os.path.splitext(output_filename)
                                    if AlignThread.detect_multiple_faces:
                                        output_img_filename_n = "{}_{}{}".format(filename_base, i, file_extension)
                                        output_dep_filename_n = "{}_{}.npy".format(filename_base, i)
                                    else:
                                        output_img_filename_n = "{}{}".format(filename_base, file_extension)
                                        output_dep_filename_n = "{}.npy".format(filename_base)
                                    misc.imsave(output_img_filename_n, scaled_img)
                                    np.save(output_dep_filename_n, scaled_dep)

                                    text_file.write('%s %d %d %d %d\n' % (output_img_filename_n, bb[0], bb[1], bb[2], bb[3]))
                            else:
                                print('Unable to align "%s"' % image_path)
                                text_file.write('%s\n' % output_img_filename_n)

        print('Total number of images: %d' % AlignThread.num_total_img)
        print('Number of successfully aligned images on Thread #%d: %d' % (self.threadID, nrof_successfully_aligned))


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_dir', type=str, help='Directory with unaligned images.', required=True)
    parser.add_argument('--output_dir', type=str, help='Directory with aligned face thumbnails.', default='./')
    parser.add_argument('--image_size', type=int,
                        help='Image size (height, width) in pixels.', default=182)
    parser.add_argument('--thread_num', type=int,
                        help='#Threads to process alignment.', default=1)
    parser.add_argument('--margin', type=int,
                        help='Margin for the crop around the bounding box (height, width) in pixels.', default=44)
    parser.add_argument('--random_order',
                        help='Shuffles the order of images to enable alignment using multiple processes.', action='store_true')
    parser.add_argument('--gpu_memory_fraction', type=float,
                        help='Upper bound on the amount of GPU memory that will be used by the process.', default=1.0)
    parser.add_argument('--detect_multiple_faces', type=bool,
                        help='Detect and align multiple faces per image.', default=False)
    return parser.parse_args(argv)


if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    AlignThread.initialization(args)
    # Multi-thread processing.
    AlignThread.threads = []
    for threadID in range(1, args.thread_num + 1):
        _thd = AlignThread(threadID)
        _thd.start()
        AlignThread.threads.append(_thd)
    for thd in AlignThread.threads:
        thd.join()
    AlignThread.pbar.close()
