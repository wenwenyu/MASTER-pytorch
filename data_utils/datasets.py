# -*- coding: utf-8 -*-
# @Author: Wenwen Yu
# @Created Time: 8/11/2020 4:49 PM

import os
import random
import io
import math
import json

from PIL import Image
import numpy as np
import lmdb

import torch
import torch.distributed as dist
from torch.utils.data import Dataset
from torch.utils.data import sampler
from torch.utils.data.sampler import Sampler
import torchvision.transforms as transforms

from utils.label_util import LabelTransformer


class TextDataset(Dataset):

    def __init__(self, txt_file=None, img_root=None, transform=None, target_transform=None, training=True, img_w=256,
                 img_h=32, case_sensitive=True, testing_with_label_file=False, convert_to_gray=True, split=','):
        '''

        :param txt_file: txt file, every line containing <ImageFile>,<Text Label>
        :param img_root:
        :param transform:
        :param target_transform:
        :param training:
        :param img_w:
        :param img_h:
        :param testing_with_label_file: if False, read image from img_root, otherwise read img from txt_file
        '''
        assert img_root is not None, 'root must be set'
        self.img_w = img_w
        self.img_h = img_h

        self.training = training
        self.case_sensitive = case_sensitive
        self.testing_with_label_file = testing_with_label_file
        self.convert_to_gray = convert_to_gray

        self.all_images = []
        self.all_labels = []

        if training or testing_with_label_file:  # for train and validation
            images, labels = get_datasets_image_label_with_txt_file(txt_file, img_root, split)
            self.all_images += images
            self.all_labels += labels
        else:  # for testing, root is image_dir
            imgs = os.listdir(img_root)
            for img in imgs:
                self.all_images.append(os.path.join(img_root, img))

        # for debug
        self.all_images = self.all_images[:]

        self.nSamples = len(self.all_images)

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        try:
            file_name = self.all_images[index]
            if self.training:
                label = self.all_labels[index]

            img = Image.open(file_name)

            try:
                if self.convert_to_gray:
                    img = img.convert('L')
                else:
                    img = img.convert('RGB')
            except Exception as e:
                print('Error Image for {}'.format(file_name))
            # exit()

            if self.transform is not None:
                img, width_ratio = self.transform(img)

            if self.target_transform is not None and self.training:
                label = self.target_transform(label)
            if self.training:
                if not self.case_sensitive:
                    label = label.lower()
                return (img, label)
            else:
                return (img, file_name)
        except Exception as read_e:
            return self.__getitem__(np.random.randint(self.__len__()))


class JSONDataset(Dataset):
    def __init__(self, txt_file=None, img_root=None, transform=None, target_transform=None, training=True, img_w=256,
                 img_h=32, case_sensitive=True, testing_with_label_file=False, convert_to_gray=True):
        '''
        :param txt_file: txt file, every line containing {ImageFile:<ImageFile>, Label:<Label>}
        :param img_root:
        :param transform:
        :param target_transform:
        :param training:
        :param img_w:
        :param img_h:
        :param testing_with_label_file: if False, read image from img_root, otherwise read img from txt_file
        '''
        assert img_root is not None, 'root must be set'
        self.img_w = img_w
        self.img_h = img_h

        self.training = training
        self.case_sensitive = case_sensitive
        self.testing_with_label_file = testing_with_label_file
        self.convert_to_gray = convert_to_gray

        self.all_images = []
        self.all_labels = []

        if training or testing_with_label_file:  # for train and validation
            images, labels = get_dataset_image_and_label_with_json_file(txt_file, img_root)
            self.all_images += images
            self.all_labels += labels
        else:  # for testing, root is image_dir
            imgs = os.listdir(img_root)
            for img in imgs:
                self.all_images.append(os.path.join(img_root, img))

        # for debug
        self.all_images = self.all_images[:]

        self.nSamples = len(self.all_images)

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        try:
            file_name = self.all_images[index]
            if self.training:
                label = self.all_labels[index]

            img = Image.open(file_name)

            try:
                if self.convert_to_gray:
                    img = img.convert('L')
                else:
                    img = img.convert('RGB')
            except Exception as e:
                print('Error Image for {}'.format(file_name))
                # exit()

            if self.transform is not None:
                img, width_ratio = self.transform(img)

            if self.target_transform is not None and self.training:
                label = self.target_transform(label)
            if self.training:
                if not self.case_sensitive:
                    label = label.lower()
                return (img, label)
            else:
                return (img, file_name)
        except Exception as read_e:
            return self.__getitem__(np.random.randint(self.__len__()))

class LmdbDataset(Dataset):
    def __init__(self, lmdb_dir_root=None, transform=None, target_transform=None, training=True, img_w=256,
                 img_h=32, case_sensitive=True,
                 convert_to_gray=True):
        self.env = lmdb.open(lmdb_dir_root,
                             max_readers=32,
                             readonly=True,
                             lock=False,
                             readahead=False,
                             meminit=False)
        if not self.env:
            raise RuntimeError('Lmdb file cannot be open')

        self.transform = transform
        self.target_transform = target_transform

        self.training = training
        self.case_sensitive = case_sensitive
        self.convert_to_gray = convert_to_gray
        self.img_w = img_w
        self.img_h = img_h

        self.image_keys, self.labels = self.__get_images_and_labels()
        self.nSamples = len(self.image_keys)

    def __len__(self):
        return self.nSamples

    def __get_images_and_labels(self):
        image_keys = []
        labels = []
        with self.env.begin(write=False) as txn:
            nSamples = int(txn.get(b"nSamples").decode())
            for i in range(nSamples):
                index = i + 1
                image_key = ('image-{}'.format(index)).encode()
                label_key = ('transcript-{}'.format(index)).encode()

                label = txn.get(label_key).decode()

                if len(label) > LabelTransformer.max_length and LabelTransformer.max_length != -1:
                    continue

                image_keys.append(image_key)
                labels.append(label)
        return image_keys, labels

    def __getitem__(self, index):
        try:
            image_key = self.image_keys[index]

            with self.env.begin(write=False) as txn:
                imgbuf = txn.get(image_key)
                buf = io.BytesIO()
                buf.write(imgbuf)
                buf.seek(0)

                try:
                    if self.convert_to_gray:
                        img = Image.open(buf).convert('L')
                    else:
                        img = Image.open(buf).convert('RGB')
                except IOError:
                    print('Error Image for {}'.format(image_key))
                    # exit()

                if self.transform is not None:
                    img, width_ratio = self.transform(img)

                label = self.labels[index]

                if self.target_transform is not None:
                    label = self.target_transform(label)

            if self.training:
                if not self.case_sensitive:
                    label = label.lower()
                return (img, label)
            else:
                return (img, image_key)
        except Exception as read_e:
            return self.__getitem__(np.random.randint(self.__len__()))


def hierarchy_dataset(root, select_data=None, training=True, img_w=256, img_h=32, transform=None, target_transform=None,
                      case_sensitive=True, convert_to_gray=True):
    """
    combine lmdb data in sub directory (MJ_train vs Synthtext)
    change sub directory if you want in config file by format "sub_dir-sub_dir-..."
    """
    dataset_list = []
    if select_data is not None:
        select_data = select_data.split('-')
        for select_d in select_data:
            dataset = LmdbVer2Dataset(lmdb_dir_root=os.path.join(root, select_d), training=training, img_w=img_w,
                                      img_h=img_h, transform=transform, target_transform=target_transform,
                                      case_sensitive=case_sensitive, convert_to_gray=convert_to_gray)
            dataset_list.append(dataset)
    concatenated_dataset = ConcatDataset(dataset_list)
    return concatenated_dataset


class LmdbVer2Dataset(Dataset):
    """
    load lmdb dataset in deep text recognition benchmark [https://github.com/clovaai/deep-text-recognition-benchmark]
    Download link: [https://www.dropbox.com/sh/i39abvnefllx2si/AABX4yjNn2iLeKZh1OAwJUffa/data_lmdb_release.zip?dl=0]
    unzip and modify folder by:
    dataset/data_lmdb_release/
        training/
            MJ_train
            ST
        val/
            MJ_test
            MJ_valid
    """
    def __init__(self, lmdb_dir_root=None, transform=None, target_transform=None, training=True, img_w=256,
                 img_h=32, case_sensitive=True,
                 convert_to_gray=True):
        self.env = lmdb.open(os.path.join(lmdb_dir_root, root),
                             max_readers=32,
                             readonly=True,
                             lock=False,
                             readahead=False,
                             meminit=False)
        if not self.env:
            raise RuntimeError('Lmdb file cannot be open')

        self.transform = transform
        self.target_transform = target_transform

        self.training = training
        self.case_sensitive = case_sensitive
        self.convert_to_gray = convert_to_gray
        self.img_w = img_w
        self.img_h = img_h

        self.image_keys, self.labels = self.__get_images_and_labels()
        self.nSamples = len(self.image_keys)

    def __len__(self):
        return self.nSamples

    def __get_images_and_labels(self):
        image_keys = []
        labels = []
        with self.env.begin(write=False) as txn:
            nSamples = int(txn.get(b"num-samples").decode())
            for i in range(nSamples):
                index = i + 1
                image_key = ('image-%09d' % index).encode()
                label_key = ('label-%09d' % index).encode()

                label = txn.get(label_key).decode()

                if len(label) > LabelTransformer.max_length and LabelTransformer.max_length != -1:
                    continue

                image_keys.append(image_key)
                labels.append(label)
        return image_keys, labels

    def __getitem__(self, index):
        try:
            image_key = self.image_keys[index]

            with self.env.begin(write=False) as txn:
                imgbuf = txn.get(image_key)
                buf = io.BytesIO()
                buf.write(imgbuf)
                buf.seek(0)

                try:
                    if self.convert_to_gray:
                        img = Image.open(buf).convert('L')
                    else:
                        img = Image.open(buf).convert('RGB')
                except IOError:
                    print('Error Image for {}'.format(image_key))
                    # exit()

                if self.transform is not None:
                    img, width_ratio = self.transform(img)

                label = self.labels[index]

                if self.target_transform is not None:
                    label = self.target_transform(label)

            if self.training:
                if not self.case_sensitive:
                    label = label.lower()
                return (img, label)
            else:
                return (img, image_key)
        except Exception as read_e:
            return self.__getitem__(np.random.randint(self.__len__()))

def get_datasets_image_label_with_txt_file(txt_file, img_root, split=','):
    image_names = []
    labels = []

    # every line containing <ImageFile,Label> text
    with open(txt_file, encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            splited = line.strip().rstrip('\n').split(split)
            image_name = splited[0]
            label = split.join(splited[1:])
            if len(label) > LabelTransformer.max_length and LabelTransformer.max_length != -1:
                continue
            image_name = os.path.join(img_root, image_name)
            image_names.append(image_name)
            labels.append(label)
    return image_names, labels


def get_dataset_image_and_label_with_json_file(txt_file, img_root):
    image_names = []
    labels = []

    # every line containing {ImageFile:<ImageFile>, Label:<Label>} json object
    with open(txt_file, encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().rstrip("\n")
            lineobj = json.loads(line)
            label = lineobj['Label']
            if len(label) > LabelTransformer.max_length and LabelTransformer.max_length != -1:
                continue
            image_name = lineobj['ImageFile']
            image_name = os.path.join(img_root, image_name)
            image_names.append(image_name)
            labels.append(label)
    return image_names, labels


class ResizeWeight(object):

    def __init__(self, size, interpolation=Image.BILINEAR, gray_format=True):
        self.w, self.h = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()
        self.gray_format = gray_format

    def __call__(self, img):
        img_w, img_h = img.size

        if self.gray_format:
            if img_w / img_h < 1.:
                img = img.resize((self.h, self.h), self.interpolation)
                resize_img = np.zeros((self.h, self.w, 1), dtype=np.uint8)
                resize_img[0:self.h, 0:self.h, 0] = img
                img = resize_img
                width = self.h
            elif img_w / img_h < self.w / self.h:
                ratio = img_h / self.h
                new_w = int(img_w / ratio)
                img = img.resize((new_w, self.h), self.interpolation)
                resize_img = np.zeros((self.h, self.w, 1), dtype=np.uint8)
                resize_img[0:self.h, 0:new_w, 0] = img
                img = resize_img
                width = new_w
            else:
                img = img.resize((self.w, self.h), self.interpolation)
                resize_img = np.zeros((self.h, self.w, 1), dtype=np.uint8)
                resize_img[:, :, 0] = img
                img = resize_img
                width = self.w

            img = self.toTensor(img)
            img.sub_(0.5).div_(0.5)
            return img, width / self.w
        else:  # RGB format
            if img_w / img_h < 1.:
                img = img.resize((self.h, self.h), self.interpolation)
                resize_img = np.zeros((self.h, self.w, 3), dtype=np.uint8)
                img = np.array(img, dtype=np.uint8)  # (w,h) -> (h,w,c)
                resize_img[0:self.h, 0:self.h, :] = img
                img = resize_img
                width = self.h
            elif img_w / img_h < self.w / self.h:
                ratio = img_h / self.h
                new_w = int(img_w / ratio)
                img = img.resize((new_w, self.h), self.interpolation)
                resize_img = np.zeros((self.h, self.w, 3), dtype=np.uint8)
                img = np.array(img, dtype=np.uint8)  # (w,h) -> (h,w,c)
                resize_img[0:self.h, 0:new_w, :] = img
                img = resize_img
                width = new_w
            else:
                img = img.resize((self.w, self.h), self.interpolation)
                resize_img = np.zeros((self.h, self.w, 3), dtype=np.uint8)
                img = np.array(img, dtype=np.uint8)  # (w,h) -> (h,w,c)
                resize_img[:, :, :] = img
                img = resize_img
                width = self.w

            img = self.toTensor(img)
            img.sub_(0.5).div_(0.5)
            return img, width / self.w


class DistCollateFn(object):
    '''
    fix bug when len(data) do not be divided by batch_size, on condition of distributed validation
    avoid error when some gpu assigned zero samples
    '''

    def __init__(self, training=True):
        self.training = training

    def __call__(self, batch):

        batch_size = len(batch)
        if batch_size == 0:
            return dict(batch_size=batch_size, images=None, labels=None)

        if self.training:
            images, labels = zip(*batch)
            image_batch_tensor = torch.stack(images, dim=0).float()
            # images Tensor: (bs, c, h, w), file_names tuple: (bs,)
            return dict(batch_size=batch_size,
                        images=image_batch_tensor,
                        labels=labels)
        else:
            images, file_names = zip(*batch)
            image_batch_tensor = torch.stack(images, dim=0).float()
            # images Tensor: (bs, c, h, w), file_names tuple: (bs,)
            return dict(batch_size=batch_size,
                        images=image_batch_tensor,
                        file_names=file_names)


class DistValSampler(Sampler):
    # DistValSampler distributes batches equally (based on batch size) to every gpu (even if there aren't enough samples)
    # This instance is used as batch_sampler args of validation dtataloader,
    # to guarantee every gpu validate different samples simultaneously
    # WARNING: Some baches will contain an empty array to signify there aren't enough samples
    # distributed=False - same validation happens on every single gpu
    def __init__(self, indices, batch_size, distributed=True):
        self.indices = indices
        self.batch_size = batch_size
        if distributed:
            self.world_size = dist.get_world_size()
            self.global_rank = dist.get_rank()
        else:
            self.global_rank = 0
            self.world_size = 1

        # expected number of batches per process. Need this so each distributed gpu validates on same number of batches.
        # even if there isn't enough data to go around
        self.expected_num_steps = math.ceil(len(self.indices) / self.world_size / self.batch_size)

        # num_samples = total samples / world_size. This is what we distribute to each gpu
        self.num_samples = self.expected_num_steps * self.batch_size

    def __iter__(self):
        current_rank_offset = self.num_samples * self.global_rank
        current_sampled_indices = self.indices[
                                  current_rank_offset:min(current_rank_offset + self.num_samples, len(self.indices))]

        for step in range(self.expected_num_steps):
            step_offset = step * self.batch_size
            # yield sampled_indices[offset:offset + self.batch_size]
            yield current_sampled_indices[step_offset:min(step_offset + self.batch_size, len(current_sampled_indices))]

    def __len__(self):
        return self.expected_num_steps

    def set_epoch(self, epoch):
        return
