import glob
from datetime import datetime
import json
import os
import random
import shutil
import jsonpickle
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

from michalski_trains.m_train import *


class MichalskiTrainDataset(Dataset):
    def __init__(self, class_rule, base_scene, raw_trains, train_vis, min_car=2, max_car=4,
                 train_count=10000, resize=False, label_noise=0, ds_path='output/image_generator'):
        """ MichalskiTrainDataset
            @param class_rule (string): classification rule
            @param base_scene (string): background scene
            @param raw_trains (string): typ of train descriptions 'RandomTrains' or 'MichalskiTrains'
            @param train_vis (string): visualization of the train description either 'MichalskiTrains' or 'SimpleObjects'
            @param train_count (int): number of train images
            @param resize: bool if true images are resized to 224x224
            @param label_noise: float between 0 and 1. If > 0, labels are flipped with the given probability
            @param ds_path: path to the dataset
            @return X_val: image data
            @return y_val: label data
            """
        self.images = []
        self.trains = []
        self.masks = []
        self.resize = resize
        self.train_count = train_count
        ds_typ = f'{train_vis}_{class_rule}_{raw_trains}_{base_scene}_len_{min_car}-{max_car}'
        self.base_scene = base_scene
        self.image_base_path = f'{ds_path}/{ds_typ}/images'
        self.all_scenes_path = f'{ds_path}/{ds_typ}/all_scenes'

        self.labels = ['direction']
        self.label_classes = ['west', 'east']
        self.class_dim = len(self.label_classes)
        self.output_dim = len(self.labels)
        # train with class specific labels
        if not os.path.isfile(self.all_scenes_path + '/all_scenes.json'):
            raise AssertionError(f'json scene file missing {self.all_scenes_path}. Not all images were generated')
        if len(os.listdir(self.image_base_path)) < self.train_count:
            raise AssertionError(f'Missing images in dataset. Expected size {self.train_count}.'
                                 f'Available images: {len(os.listdir(self.image_base_path))}')

        path = self.all_scenes_path + '/all_scenes.json'
        with open(path, 'r') as f:
            all_scenes = json.load(f)
            for scene in all_scenes['scenes'][:train_count]:
                self.images.append(scene['image_filename'])
                # self.depths.append(scene['depth_map_filename'])
                train = scene['m_train']
                # self.trains.append(
                #     train.replace('michalski_trains.m_train.', 'm_train.'))
                self.trains.append(jsonpickle.decode(train))
                self.masks.append(scene['car_masks'])

        trans = [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ]
        if resize:
            print('resize true')
            trans.append(transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC))
        self.norm = transforms.Compose(trans)
        self.normalize_mask = transforms.Compose([
            # transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5]),
        ])

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


def get_datasets(base_scene='base_scene', raw_trains='MichalskiTrains', train_vis='Trains', min_car=2, max_car=4,
                 ds_size=12000, ds_path='output/image_generator', class_rule='theoryx', resize=False, noise=0):
    """
    Returns the train and validation dataset for the given parameters
    Args:
        @param base_scene: string scene to be used for the dataset
        @param raw_trains: train description to be used for the dataset
        @param train_vis: train visualization to be used for the dataset
        @param min_car: minimum number of cars in the scene
        @param max_car: maximum number of cars in the scene
        @param ds_size: dataset size
        @param ds_path: path to the dataset
        @param class_rule: class rule to be used for the dataset
        @param resize: whether to resize the images to 224x224
        @param noise: whether to apply noise to the labels
    @return: michalski train dataset
    """
    path_ori = f'{ds_path}/{train_vis}_{class_rule}_{raw_trains}_{base_scene}_len_2-4'
    if not os.path.isfile(path_ori + '/all_scenes/all_scenes.json'):
        combine_json(base_scene, raw_trains, train_vis, class_rule, ds_size=ds_size)
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
    full_ds = MichalskiTrainDataset(class_rule=class_rule, base_scene=base_scene, raw_trains=raw_trains,
                                    train_vis=train_vis, min_car=min_car, max_car=max_car,
                                    label_noise=noise, train_count=ds_size, resize=resize, ds_path=ds_path)
    return full_ds


def combine_json(path_settings, out_dir='output/image_generator', ds_size=10000):
    path_ori = f'tmp/image_generator/{path_settings}'
    path_dest = f'{out_dir}/{path_settings}'
    im_path = path_ori + '/images'
    if os.path.isdir(im_path):
        files = os.listdir(im_path)
        if len(files) == ds_size:
            merge_json_files(path_ori)
            shutil.rmtree(path_ori + '/scenes')
            try:
                shutil.rmtree(path_dest)
            except:
                pass
            shutil.move(path_ori, path_dest)


def merge_json_files(path):
    """
    merging all ground truth json files of the dataset
    :param:  path (string)        : path to the dataset information
    """
    all_scenes = []
    for p in glob.glob(path + '/scenes/*_m_train.json'):
        with open(p, 'r') as f:
            all_scenes.append(json.load(f))
    output = {
        'info': {
            'date': datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
            'version': '0.1',
            'license': None,
        },
        'scenes': all_scenes
    }
    json_pth = path + '/all_scenes/all_scenes.json'
    os.makedirs(path + '/all_scenes/', exist_ok=True)
    # args.output_scene_file.split('.json')[0]+'_classid_'+str(args.img_class_id)+'.json'
    with open(json_pth, 'w+') as f:
        json.dump(output, f, indent=2)
