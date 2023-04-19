import json
import json
import os
import random

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from michalski_trains.m_train import *


class MichalskiDataset(Dataset):
    def __init__(self, class_rule, base_scene, raw_trains, train_vis, min_car=2, max_car=4,
                 ds_size=10000, resize=False, label_noise=0, image_noise=0, ds_path='output/image_generator',
                 preprocessing=None):
        """ MichalskiTrainDataset
            @param class_rule (string): classification rule
            @param base_scene (string): background scene
            @param raw_trains (string): typ of train descriptions 'RandomTrains' or 'MichalskiTrains'
            @param train_vis (string): visualization of the train description either 'MichalskiTrains' or 'SimpleObjects'
            @param train_count (int): number of train images
            @param resize: bool if true images are resized to 224x224
            @param label_noise: float between 0 and 1. If > 0, labels are flipped with the given probability
            @param ds_path: path to the dataset
            @param preprocessing: preprocessing function to apply to the images
            @return X_val: image data
            @return y_val: label data
            """
        # ds data
        self.images, self.trains, self.masks = [], [], []
        # ds settings
        self.class_rule, self.base_scene, self.raw_trains, self.train_vis = class_rule, base_scene, raw_trains, train_vis
        self.min_car, self.max_car = min_car, max_car
        self.resize, self.train_count, self.label_noise = resize, ds_size, label_noise
        # ds path
        self.ds_typ = f'{train_vis}_{class_rule}_{raw_trains}_{base_scene}_len_{min_car}-{max_car}'
        self.image_base_path = f'{ds_path}/{self.ds_typ}/images'
        self.all_scenes_path = f'{ds_path}/{self.ds_typ}/all_scenes'

        # ds labels
        self.labels = ['direction']
        self.label_classes = ['west', 'east']
        self.attributes = ['color', 'length', 'walls', 'roofs', 'wheel_count', 'load_obj1', 'load_obj2',
                           'load_obj3'] * 4
        color = ['yellow', 'green', 'grey', 'red', 'blue']
        length = ['short', 'long']
        walls = ["braced_wall", 'solid_wall']
        roofs = ["roof_foundation", 'solid_roof', 'braced_roof', 'peaked_roof']
        wheel_count = ['2_wheels', '3_wheels']
        load_obj = ["box", "golden_vase", 'barrel', 'diamond', 'metal_pot', 'oval_vase']
        self.attribute_classes = ['none'] + color + length + walls + roofs + wheel_count + load_obj
        # train with class specific labels
        if not os.path.isfile(self.all_scenes_path + '/all_scenes.json'):
            raise AssertionError(f'json scene file missing {self.all_scenes_path}. Not all images were generated')
        if len(os.listdir(self.image_base_path)) < self.train_count:
            raise AssertionError(f'Missing images in dataset. Expected size {self.train_count}.'
                                 f'Available images: {len(os.listdir(self.image_base_path))}')

        # load data
        path = self.all_scenes_path + '/all_scenes.json'
        with open(path, 'r') as f:
            all_scenes = json.load(f)
            for scene in all_scenes['scenes'][:ds_size]:
                self.images.append(scene['image_filename'])
                # self.depths.append(scene['depth_map_filename'])
                train = scene['m_train']
                # self.trains.append(
                #     train.replace('michalski_trains.m_train.', 'm_train.'))
                self.trains.append(jsonpickle.decode(train))
                self.masks.append(scene['car_masks'])
        # transform

        self.image_size = self.get_image_size(0)

        trans = [transforms.ToTensor()]
        if preprocessing is not None:
            trans.append(preprocessing)
        if resize:
            print('resize true')
            trans.append(transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC))
        if image_noise > 0:
            print('adding noise to images')
            trans.append(AddBinaryNoise(image_noise))

        trans.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))

        self.norm = transforms.Compose(trans)
        self.normalize_mask = transforms.Compose([
            # transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5]),
        ])
        # add noise to labels
        if label_noise > 0:
            print(f'applying noise of {label_noise} to dataset labels')
            for train in self.trains:
                n = random.random()
                if n < label_noise:
                    lab = train.get_label()
                    if lab == 'east':
                        train.set_label('west')
                    elif lab == 'west':
                        train.set_label('east')
                    else:
                        raise ValueError(f'unexpected label value {lab}, expected value east or west')

    def __getitem__(self, item):
        image = self.get_pil_image(item)
        X = self.norm(image)
        y = self.get_direction(item)
        return X, y

    def __len__(self):
        return self.train_count

    def get_image_size(self, item):
        im = self.get_pil_image(item)
        return im.size

    def get_direction(self, item):
        lab = self.trains[item].get_label()
        if lab == 'none':
            # return torch.tensor(0).unsqueeze(dim=0)
            raise AssertionError(f'There is no direction label for the selected DS {self.ds_typ}')
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

    def get_image(self, item):
        im = self.get_pil_image(item)
        return self.norm(im)

    def get_image_path(self, item):
        return self.image_base_path + '/' + self.images[item]

    def get_label_for_id(self, item):
        return self.trains[item].get_label()

    def get_trains(self):
        return self.trains

    def get_ds_labels(self):
        return self.labels

    def get_ds_classes(self):
        return self.label_classes

    def get_class_dim(self):
        return len(self.label_classes)

    def get_output_dim(self):
        return len(self.labels)


class AddBinaryNoise(object):
    def __init__(self, p=0.1):
        self.p = p

    def __call__(self, tensor):
        t = torch.ones_like(tensor)
        t[torch.rand_like(tensor) < self.p] = 0
        return t * tensor

    def __repr__(self):
        return self.__class__.__name__ + '(percentage={0})'.format(self.p)


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)