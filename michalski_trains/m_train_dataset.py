import json
import os
import shutil
from distutils.dir_util import copy_tree
from pycocotools import mask as maskUtils

import jsonpickle
import numpy as np
import torch
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from timm.data.parsers.parser import Parser
from torchvision.transforms import InterpolationMode
from tqdm.contrib import tenumerate

from blender_image_generator.json_util import merge_json_files, combine_json


class MichalskiTrainDataset(Dataset):
    def __init__(self, base_scene, raw_trains, train_vis, train_count=10000, resize=False,
                 ds_path='output/image_generator',
                 ):
        """ MichalskiTrainDataset
            Args:
                val: bool if model is used for vaildation
                resize: bool if true images are resized to 224x224
                :param:  raw_trains (string): typ of train descriptions 'RandomTrains' or 'MichalskiTrains'
                :param:  train_vis (string): visualization of the train description either 'MichalskiTrains' or
                'SimpleObjects'
            @return:
                X_val: X value output for training data returned in __getitem__()
                ['image', 'predicted_attributes', 'gt_attributes', 'gt_attributes_individual_class', 'predicted_mask', gt_mask]
                        image (torch): image of michalski train

                y_val: ['direction','attribute','mask'] y label output for training data returned in __getitem__()

            """
        self.images = []
        self.trains = []
        self.masks = []
        self.resize = resize
        self.train_count = train_count
        ds_typ = f'{raw_trains}/{train_vis}/{base_scene}'
        self.base_scene = base_scene
        self.image_base_path = f'{ds_path}/{ds_typ}/images'
        self.all_scenes_path = f'{ds_path}/{ds_typ}/all_scenes'

        self.labels = ['direction']
        self.label_classes = ['west', 'east']
        self.class_dim = len(self.label_classes)
        self.output_dim = len(self.labels)
        # train with class specific labels
        if not os.path.isfile(self.all_scenes_path + '/all_scenes.json'):
            raise AssertionError('json scene file missing. Not all images were generated')
        if len(os.listdir(self.image_base_path)) < self.train_count:
            raise AssertionError(f'Missing images in dataset. Expected size {self.train_count}.'
                                 f'Available images: {len(os.listdir(self.image_base_path))}')

        path = self.all_scenes_path + '/all_scenes.json'
        with open(path, 'r') as f:
            all_scenes = json.load(f)
            for scene in all_scenes['scenes'][:train_count]:
                self.images.append(scene['image_filename'])
                # self.depths.append(scene['depth_map_filename'])
                train = jsonpickle.decode(scene['m_train'])
                self.trains.append(train)
                self.masks.append(scene['car_masks'])

        trans = [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ]
        if resize:
            trans.append(transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC))
        self.norm = transforms.Compose(trans)
        self.normalize_mask = transforms.Compose([
            # transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5]),
        ])

    def __getitem__(self, item):

        image = self.get_pil_image(item)
        X = self.norm(image)
        y = self.get_direction(item)
        return X, y

    def __len__(self):
        return self.train_count

    def get_direction(self, item):
        lab = self.trains[item].get_label()
        if lab == 'none':
            # return torch.tensor(0).unsqueeze(dim=0)
            raise AssertionError(f'There is no direction label for a RandomTrains. Use MichalskiTrain DS.')
        label_binary = self.label_classes.index(lab)
        label = torch.tensor(label_binary).unsqueeze(dim=0)
        return label

    def get_m_train(self, item):
        return self.trains[item]

    def get_mask(self, item):
        return self.masks[item]

    def get_pil_image(self, item):
        im_path = self.get_image_path(item)
        return Image.open(im_path).convert('RGB')

    def get_image_path(self, item):
        return self.image_base_path + '/' + self.images[item]

    def get_label_for_id(self, item):
        return self.trains[item].get_label()

    def get_trains(self):
        return self.trains

    def get_ds_labels(self):
        return self.labels



def get_datasets(base_scene, raw_trains, train_vis, ds_size, resize=False, ds_path='output/image_generator'):
    path_ori = f'{ds_path}/{raw_trains}/{train_vis}/{base_scene}'
    if not os.path.isfile(path_ori + '/all_scenes/all_scenes.json'):
        combine_json(base_scene, raw_trains, train_vis, ds_size)
        raise Warning(f'Dataloader did not find JSON ground truth information.'
                      f'Might be caused by interruptions during process of image generation.'
                      f'Generating new JSON file at: {path_ori + "/all_scenes/all_scenes.json"}')
    im_path = path_ori + '/images'
    if not os.path.isdir(im_path):
        raise AssertionError('dataset not found, please generate images first')

    files = os.listdir(im_path)
    # total image count equals 10.000 adjust if not all images need to be generated
    if len(files) < ds_size:
        raise AssertionError(
            f'not enough images in dataset: expected {ds_size}, present: {len(files)}'
            f' please generate the missing images')
    elif len(files) > ds_size:
        raise Warning(
            f' dataloader did not select all images of the dataset, number of selected images:  {ds_size},'
            f' available images in dataset: {len(files)}')

    # merge json files to one if it does not already exist
    if not os.path.isfile(path_ori + '/all_scenes/all_scenes.json'):
        raise AssertionError(
            f'no JSON found')
    # image_count = None for standard image count
    full_ds = MichalskiTrainDataset(base_scene=base_scene, raw_trains=raw_trains, train_vis=train_vis,
                                    train_count=ds_size, resize=resize, ds_path=ds_path)
    return full_ds
