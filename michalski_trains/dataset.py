import glob
import warnings
from datetime import datetime
import json
import os
import random
import shutil
import jsonpickle
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from pycocotools import mask as maskUtils

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
        ds_typ = f'{train_vis}_{class_rule}_{raw_trains}_{base_scene}_len_{min_car}-{max_car}'
        self.image_base_path = f'{ds_path}/{ds_typ}/images'
        self.all_scenes_path = f'{ds_path}/{ds_typ}/all_scenes'

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

    def get_ds_classes(self):
        return self.label_classes

    def get_class_dim(self):
        return len(self.label_classes)

    def get_output_dim(self):
        return len(self.labels)


class MichalskiAttributeDataset(MichalskiDataset):
    ''' Dataset for the Michalski train dataset with attributes as labels.
        parameters:
        fixed_output_car_size: if set to None, the output tensor will have the size of the number of cars * attributes.
         Else the output tensor y will have the size of fixed_output_car_size * number of attributes
    '''

    def __int__(self, fixed_car_size=4, **kwargs):
        self.fixed_car_size = fixed_car_size
        super().__init__(**kwargs)

    def __getitem__(self, item):
        image = self.get_pil_image(item)
        X = self.norm(image)
        # y = self.get_attributes(item) if self.fixed_output_car_size is None else self.get_attributes_fixed_size(item)
        y = self.get_attributes_fixed_size(item)
        return X, y

    def get_attributes(self, item):
        ''' Returns a tensor of labels for all attributes of each car in the train
        :param item: index of the train
        :return: tensor of labels
        '''
        att = self.attribute_classes
        train = self.trains[item]
        cars = train.get_cars()
        labels = []
        # each train has (n cars a 8 attributes) totalling to n*8 labels
        # each label can have 22 classes
        for car in cars:
            # index 0 = not existent
            color = att.index(car.get_blender_car_color())
            length = att.index(car.get_car_length())
            wall = att.index(car.get_blender_wall())
            roof = att.index(car.get_blender_roof())
            wheels = att.index(car.get_car_wheels())
            l_shape = att.index(car.get_blender_payload())
            l_num = car.get_load_number()
            l_shapes = [l_shape] * l_num + [0] * (3 - l_num)
            car_number = car.get_car_number()
            labels += [color, length, wall, roof, wheels] + l_shapes
        return torch.tensor(labels, dtype=torch.int64)

    def get_attributes_fixed_size(self, item):
        ''' Returns a tensor of labels for all attributes of each car in the train
        :param item: index of the train
        :return: tensor of labels
        '''
        # fixed_car_size = self.fixed_car_size
        fixed_car_size = 4
        # self.fixed_output_car_size = 4
        att = self.attribute_classes
        train = self.trains[item]
        cars = train.get_cars()
        labels = []
        # each train has (fixed_output_car_size cars a 8 attributes) totalling to fixed_output_car_size*8 labels
        # each label can have 22 classes
        for car in cars:
            # index 0 = not existent
            color = att.index(car.get_blender_car_color())
            length = att.index(car.get_car_length())
            wall = att.index(car.get_blender_wall())
            roof = att.index(car.get_blender_roof())
            wheels = att.index(car.get_car_wheels())
            l_shape = att.index(car.get_blender_payload())
            l_num = car.get_load_number()
            l_shapes = [l_shape] * l_num + [0] * (3 - l_num)
            car_number = car.get_car_number()
            labels += [color, length, wall, roof, wheels] + l_shapes
        if len(labels) > fixed_car_size * 8:
            raise ValueError(f'Number of labels is greater than the fixed output car size.')
        labels = labels + [0] * (fixed_car_size * 8 - len(labels))
        return torch.tensor(labels, dtype=torch.int64)

    def get_ds_labels(self):
        return self.attributes

    def get_ds_classes(self):
        return self.attribute_classes

    def get_class_dim(self):
        return len(self.attribute_classes)

    def get_output_dim(self):
        return len(self.attributes)


class MichalskiMaskDataset(MichalskiAttributeDataset):
    # def __int__(self, *args, **kwargs):
    # super().__init__(*args, **kwargs)

    def __getitem__(self, item):
        target = {}
        image = self.get_pil_image(item)
        X = self.norm(image)
        boxes = self.get_bboxes(item)
        labels, label_ids = self.get_mask_labels(item)
        masks = self.get_masks(item)

        target['boxes'] = boxes
        target['labels'] = labels
        target['labels_ids'] = label_ids
        target['image_id'] = torch.tensor([item], dtype=torch.int64)
        target['area'] = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        target['iscrowd'] = torch.zeros_like(target['area'], dtype=torch.uint8)
        # not sure how crowd is to be defined in this dataset
        # target['iscrowd'] = torch.zeros((boxes.shape[0],), dtype=torch.int64)
        target['masks'] = masks
        return X, target

    def get_mask_labels(self, item):
        ''' Returns a tensor of labels for all attributes of each car in the train
        :param item: index of the train
        :return: tensor of labels
        '''
        att = self.attribute_classes
        train = self.trains[item]
        cars = train.get_cars()
        labels = []
        label_ids = []
        # each train has (n cars a 9 attributes) totalling to n*9 labels
        # each label can have 22 classes + n for total number of cars
        for car in cars:
            # index 0 = not existent
            color = att.index(car.get_blender_car_color())
            length = att.index(car.get_car_length())
            wall = att.index(car.get_blender_wall())
            roof = att.index(car.get_blender_roof())
            wheels = att.index(car.get_car_wheels())
            l_shape = att.index(car.get_blender_payload())
            l_num = car.get_load_number()
            l_shapes = [l_shape] * l_num + [0] * (3 - l_num)
            car_number_label = car.get_car_number() + len(att) - 1
            labels += [car_number_label, color, length, wall, roof, wheels] + l_shapes
            # label_ids += [0, 1, 2, 3, 4, 5, 6, 6, 6]
            # label_ids += [0, 2, 1, 1, 1, 1, 1, 1, 1]
            label_ids += [0, 1, 1, 1, 1, 1, 1, 1, 1]
        # remove the 0 labels (not existent)
        labels = torch.tensor(labels, dtype=torch.int64)
        label_ids = torch.tensor(label_ids, dtype=torch.int64)
        return labels[labels != 0], label_ids[labels != 0]

    def get_mask_label_ids(self, item):
        ''' Returns a tensor of labels for all attributes of each car in the train
        :param item: index of the train
        :return: tensor of labels
        '''
        att = self.attribute_classes
        train = self.trains[item]
        cars = train.get_cars()
        labels = []
        # each train has (n cars a 9 attributes) totalling to n*9 labels
        # each label can have 22 classes + n for total number of cars
        for car in cars:
            # index 0 = not existent
            color = att.index(car.get_blender_car_color())
            length = att.index(car.get_car_length())
            wall = att.index(car.get_blender_wall())
            roof = att.index(car.get_blender_roof())
            wheels = att.index(car.get_car_wheels())
            l_shape = att.index(car.get_blender_payload())
            l_num = car.get_load_number()
            l_shapes = [l_shape] * l_num + [0] * (3 - l_num)
            car_number_label = car.get_car_number() + len(att) - 1
            labels += [car_number_label, color, length, wall, roof, wheels] + l_shapes
        # remove the 0 labels (not existent)
        labels = torch.tensor(labels, dtype=torch.int64)
        return labels[labels != 0]

    def get_masks(self, item):
        ''' returns a list of masks of the train. Each mask is a tensor of shape (270, 480).
        It returns 9 masks for each car in the train. the corresponding labels are:
        0: whole car
        1: color
        2: length
        3: wall
        4: roof
        5: wheels
        6: payload_0
        7: payload_1
        8: payload_2
        params:
            item: index of the train
        returns:
            tensor of masks of shape (9*car numbers, 270, 480), dtype = torch.uint8
        '''
        w, h = self.image_size
        masks = torch.empty(0, h, w, dtype=torch.uint8)
        mask = self.get_mask(item)
        attr_id = -1
        y = self.get_attributes(item)
        for car_id, car in mask.items():
            whole_car_mask = car['mask']
            masks = torch.vstack([masks, torch.from_numpy(maskUtils.decode(whole_car_mask)).unsqueeze(0)])
            for att_name in ['color', 'length', 'wall', 'roof', 'wheels', 'payload_0', 'payload_1', 'payload_2']:
                attr_id += 1
                if att_name in car:
                    att = car[att_name]
                    label = att['label']
                    if label != 'none':
                        if att_name == 'length' or att_name == 'color':
                            rle = whole_car_mask
                        else:
                            rle = att['mask']
                        masks = torch.vstack([masks, torch.from_numpy(maskUtils.decode(rle)).unsqueeze(0)])
                        if y[attr_id] == 0:
                            raise AssertionError(
                                att_name + f' mask and label inconsistency in car {car_id} of train {item}')
                    else:
                        # if object is not in image add a zero mask
                        # masks = torch.vstack([masks, torch.zeros(1, 270, 480, dtype=torch.uint8)])
                        if y[attr_id] != 0:
                            raise AssertionError(
                                att_name + f' mask and label inconsistency in car {car_id} of train {item}')
                else:
                    # if object is not in image add a zero mask
                    # masks = torch.vstack([masks, torch.zeros(1, 270, 480, dtype=torch.uint8)])
                    if y[attr_id] != 0:
                        raise AssertionError(
                            att_name + f' mask and label inconsistency in car {car_id} of train {item}')
        # masks = masks[masks != 0].view(-1, h, w)
        return masks

    def get_rle(self, item):
        ''' returns a list of masks of the train. Each mask is a tensor of shape (270, 480).
        It returns 9 masks for each car in the train. the corresponding labels are:
        0: whole car
        1: color
        2: length
        3: wall
        4: roof
        5: wheels
        6: payload_0
        7: payload_1
        8: payload_2
        params:
            item: index of the train
        returns:
            mask: rle of the masks
        '''
        w, h = self.image_size
        masks = []
        mask = self.get_mask(item)
        attr_id = -1
        y = self.get_attributes(item)
        for car_id, car in mask.items():
            whole_car_mask = car['mask']
            masks.append(whole_car_mask)
            for att_name in ['color', 'length', 'wall', 'roof', 'wheels', 'payload_0', 'payload_1', 'payload_2']:
                attr_id += 1
                if att_name in car:
                    att = car[att_name]
                    label = att['label']
                    if label != 'none':
                        if att_name == 'length' or att_name == 'color':
                            rle = whole_car_mask
                        else:
                            rle = att['mask']
                        masks.append(rle)
                        if y[attr_id] == 0:
                            raise AssertionError(
                                att_name + f' mask and label inconsistency in car {car_id} of train {item}')
                    else:
                        # if object is not in image add a zero mask
                        # masks = torch.vstack([masks, torch.zeros(1, 270, 480, dtype=torch.uint8)])
                        if y[attr_id] != 0:
                            raise AssertionError(
                                att_name + f' mask and label inconsistency in car {car_id} of train {item}')
                else:
                    # if object is not in image add a zero mask
                    # masks = torch.vstack([masks, torch.zeros(1, 270, 480, dtype=torch.uint8)])
                    if y[attr_id] != 0:
                        raise AssertionError(
                            att_name + f' mask and label inconsistency in car {car_id} of train {item}')
        # masks = masks[masks != 0].view(-1, h, w)
        return masks

    def get_bboxes(self, item, format='[x0,y0,x1,y1]'):
        ''' returns a list of bounding boxes of the train. Each bounding box is a tensor of shape (270, 480).
        It returns 9 bounding boxes for each car in the train. the corresponding labels are:
        0: whole car
        1: color
        2: length
        3: wall
        4: roof
        5: wheels
        6: payload_0
        7: payload_1
        8: payload_2
        params:
            item: index of the train
            format: format of the bboxes either [x0,y0,w,h] or [x0,y0,x1,y1]
        returns:
            FloatTensor of bounding boxes of shape (9*car numbers, 270, 480)
        '''
        bboxes = torch.empty(0, 4)
        mask = self.get_mask(item)
        attr_id = -1
        y = self.get_attributes(item)
        for car_id, car in mask.items():
            whole_car_mask = car['mask']
            whole_car_bbox = maskUtils.toBbox(whole_car_mask)
            box_formated = (whole_car_bbox + np.concatenate(
                ([0, 0], whole_car_bbox[:2]))) if format == '[x0,y0,x1,y1]' else whole_car_bbox
            bboxes = torch.vstack([bboxes, torch.tensor(box_formated)])
            for att_name in ['color', 'length', 'wall', 'roof', 'wheels', 'payload_0', 'payload_1', 'payload_2']:
                attr_id += 1
                if att_name in car:
                    att = car[att_name]
                    label = att['label']
                    if label != 'none':
                        if att_name == 'length' or att_name == 'color':
                            box = whole_car_bbox
                        else:
                            box = maskUtils.toBbox(att['mask'])
                        box_formated = (box + np.concatenate(([0, 0], box[:2]))) if format == '[x0,y0,x1,y1]' else box
                        bboxes = torch.vstack([bboxes, torch.tensor(box_formated)])
                        if y[attr_id] == 0:
                            raise AssertionError(
                                att_name + f' mask and label inconsistency in car {car_id} of train {item}')
                    else:
                        bboxes = torch.vstack([bboxes, torch.zeros(1, 4)])
                        if y[attr_id] != 0:
                            raise AssertionError(
                                att_name + f' mask and label inconsistency in car {car_id} of train {item}')
                else:
                    # if object is not in image, add a zero bbox
                    bboxes = torch.vstack([bboxes, torch.zeros(1, 4)])
                    if y[attr_id] != 0:
                        raise AssertionError(
                            att_name + f' mask and label inconsistency in car {car_id} of train {item}')
        # remove zero bboxes (not present in the image)
        bboxes = bboxes[bboxes != 0].view(-1, 4)
        return bboxes


def get_datasets(base_scene, raw_trains, train_vis, class_rule, min_car=2, max_car=4,
                 ds_size=12000, ds_path='output/image_generator', y_val='direction', resize=False, label_noise=0,
                 image_noise=0, preprocessing=None, fixed_output_car_size=4):
    """
    Returns the train and validation dataset for the given parameters
    Args:
        @param base_scene: string scene to be used for the dataset
        @param raw_trains: train description to be used for the dataset
        @param train_vis: train visualization to be used for the dataset
        @param class_rule: class rule to be used for the dataset
        @param min_car: minimum number of cars in the scene
        @param max_car: maximum number of cars in the scene
        @param ds_size: dataset size
        @param ds_path: path to the dataset
        @param y_val: which value to use for the y value, either direction or attributes
        @param resize: whether to resize the images to 224x224
        @param label_noise: noise to be applied to the labels
        @param image_noise: noise to be applied to the images
        @param preprocessing: additional preprocessing to be applied to the images
    @return: michalski train dataset
    """
    path_settings = f'{train_vis}_{class_rule}_{raw_trains}_{base_scene}_len_{min_car}-{max_car}'
    print(f'set up dataset: {path_settings}')
    check_data(ds_path, path_settings, ds_size)

    # image_count = None for standard image count
    if y_val == 'direction':
        full_ds = MichalskiDataset(class_rule=class_rule, base_scene=base_scene, raw_trains=raw_trains,
                                   train_vis=train_vis, min_car=min_car, max_car=max_car,
                                   label_noise=label_noise, ds_size=ds_size, resize=resize, ds_path=ds_path,
                                   image_noise=image_noise, preprocessing=preprocessing)
    elif y_val == 'attributes':
        full_ds = MichalskiAttributeDataset(class_rule=class_rule, base_scene=base_scene, raw_trains=raw_trains,
                                            train_vis=train_vis, min_car=min_car, max_car=max_car,
                                            label_noise=label_noise, ds_size=ds_size, resize=resize, ds_path=ds_path,
                                            image_noise=image_noise, preprocessing=preprocessing)
    elif y_val == 'mask':
        full_ds = MichalskiMaskDataset(class_rule=class_rule, base_scene=base_scene, raw_trains=raw_trains,
                                       train_vis=train_vis, min_car=min_car, max_car=max_car,
                                       label_noise=label_noise, ds_size=ds_size, resize=resize, ds_path=ds_path,
                                       image_noise=image_noise, preprocessing=preprocessing)
    else:
        raise AssertionError(f'Unknown y value {y_val}')
    return full_ds


def combine_json(path_settings, out_dir='output/image_generator', ds_size=10000):
    path_ori = f'output/tmp/image_generator/{path_settings}'
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


def check_data(ds_path, path_settings, ds_size):
    path_ori = f'{ds_path}/{path_settings}'
    if not os.path.isfile(path_ori + '/all_scenes/all_scenes.json'):
        combine_json(path_settings, out_dir=ds_path, ds_size=ds_size)
        warnings.warn(f'Dataloader did not find JSON ground truth information.'
                      f'Might be caused by interruptions during process of image generation.'
                      f'Generating new JSON file at: {path_ori + "/all_scenes/all_scenes.json"}')
    im_path = path_ori + '/images'
    if not os.path.isdir(im_path):
        raise AssertionError(f'dataset not found, please generate images first: ({im_path})')

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


def michalski_categories():
    color = ['yellow', 'green', 'grey', 'red', 'blue']
    length = ['short', 'long']
    walls = ["braced_wall", 'solid_wall']
    roofs = ["roof_foundation", 'solid_roof', 'braced_roof', 'peaked_roof']
    wheel_count = ['2_wheels', '3_wheels']
    load_obj = ["box", "golden_vase", 'barrel', 'diamond', 'metal_pot', 'oval_vase']
    return ['none'] + color + length + walls + roofs + wheel_count + load_obj


def michalski_labels():
    return ['color', 'length', 'walls', 'roofs', 'wheel_count', 'load_obj1', 'load_obj2', 'load_obj3']


def rcnn_michalski_categories():
    cat = michalski_categories()
    cat += [f'car_{i}' for i in range(1, 21)]
    return cat


def rcnn_michalski_labels():
    return ['car_number', 'color', 'length', 'walls', 'roofs', 'wheel_count', 'load_obj']
