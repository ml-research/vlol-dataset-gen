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

from blender_image_generator.json_util import merge_json_files


class MichalskiTrainDataset(Dataset):
    def __init__(self, base_scene, raw_trains, train_vis, train_count=10000, val=False, resize=False,
                 X_val='image', y_val='direction'):
        """ MichalskiTrainDataset
            Args:
                val: bool if model is used for vaildation
                resize: bool if true images are resized to 224x224
                :param:  raw_trains (string): typ of train descriptions 'RandomTrains' or 'MichalskiTrains'
                :param:  train_vis (string): visualization of the train description either 'MichalskiTrains' or
                'SimpleObjects'
                X_val: X value output for training data returned in __getitem__()
                ['image', 'predicted_attributes', 'gt_attributes', 'gt_attributes_individual_class', 'predicted_mask', gt_mask]
                        image (torch): image of michalski train
                        gt_attributes: ground truth attributes of michalski train (each attribute uses general classes)
                        predicted_attributes: from attrNet/ResNet predicted attributes (each attribute uses general classes)
                        predicted_attributes_permutation_invariant: from attrNet/ResNet predicted attributes
                        (car position encoded in classes and remove everything not existent -> permutation-invariant set)
                        gt_attributes_individual_class: ground truth attributes of michalski train (each attribute has
                            its individual classes -> different number of classes for each label)

                        predicted_mask: from detectron predicted masks of michalski train multiplied with the predicted
                            class label and normalized by class count
                        gt_masks: ground truth masks for all michalski train attributes multiplied with their
                            corresponding class label label and normalized by class count
                        gt_single_mask: ground truth of a single mask multiplied by the corresponding class
                            (normalized by class count) and concatenated with the input image

                        gt_positions: ground truth position for all michalski train attributes concatenated with their
                            labels (label,x,y,z)
                        pred_positions: predicted position for all michalski train attributes concatenated with their
                            predicted labels (label,x,y,z)

                y_val: ['direction','attribute','mask'] y label output for training data returned in __getitem__()
                        direction: direction of michalski train (eastbound or westbound)
                        attribute: attributes of michalski train (each attribute uses general classes)
                        attribute_binary: attributes of michalski train (each attribute uses general classes)
                            converted to binary array of 0 and 1 (32 outputs * 22 classes = 704 values)
                        attribute_individual_class: attributes of michalski train (each attribute has its individual
                            classes)
                        mask: masks of michalski train
                        gt_position: ground truth position for a single mask

            """
        self.images = []
        self.depths = []
        self.trains = []
        self.masks = []
        self.predictions = []
        self.val = val
        self.resize = resize
        self.train_count = train_count
        self.y_val = y_val
        self.X_val = X_val
        ds_typ = f'{raw_trains}/{train_vis}/{base_scene}/'
        self.base_scene = base_scene
        self.image_base_path = f'dataset/{ds_typ}/images'
        self.scene_base_path = f'dataset/{ds_typ}/scenes'
        self.all_scenes_path = f'dataset/{ds_typ}/all_scenes'
        self.predictions_path = f'output/detectron/{ds_typ}/predictions.json'
        self.predictions_coords_path = f'output/predictions/{ds_typ}/world_coord_predictions.npy'
        self.predictions_im_count = 8000

        color = ['yellow', 'green', 'grey', 'red', 'blue']
        length = ['short', 'long']
        walls = ["braced_wall", 'solid_wall']
        roofs = ["roof_foundation", 'solid_roof', 'braced_roof', 'peaked_roof']
        wheel_count = ['2_wheels', '3_wheels']
        load_obj = ["box", "golden_vase", 'barrel', 'diamond', 'metal_pot', 'oval_vase']  # correct order
        load_obj = ['diamond', "box", "golden_vase", 'barrel', 'metal_pot', 'oval_vase']

        # a total of 22 atributes
        # load number is not in attributes since the load objects are segmented individually
        self.attributes = ['color', 'length', 'wall', 'roof', 'wheels', 'load_obj']
        self.attribute_classes = ['none'] + color + length + walls + roofs + wheel_count + load_obj
        self.classes_per_attribute = [color, length, walls, roofs, wheel_count, load_obj]
        self.direction_classes = ['west', 'east']

        if y_val == 'direction':
            # 1 label per train
            # binary label classification
            self.labels = ['direction']
            self.label_classes = self.direction_classes
            self.classes_per_label = [self.label_classes]

        elif 'position' in y_val:
            self.labels = ['x', 'y', 'z']
            self.label_classes = [None]
            self.classes_per_label = None
        else:
            l_num = [0, 1, 2, 3]
            # 32 labels per train
            # train with general classes 22 classes per label
            self.labels = ['color', 'length', 'wall', 'roof', 'wheels', 'load_1', 'load_2', 'load_3'] * 4
            self.label_classes = self.attribute_classes
            self.classes_per_label = [color, length, walls, roofs, wheel_count, load_obj, load_obj, load_obj] * 4
        self.class_dim = len(self.label_classes)
        self.output_dim = len(self.labels)
        # train with class specific labels
        # self.labels = ['color', 'length', 'wall', 'roof', 'wheels', 'l_num', 'l_shape'] * 4
        if not os.path.isfile(self.all_scenes_path + '/all_scenes.json'):
            raise AssertionError('json scene file missing. Not all images were generated')
        if len(os.listdir(self.image_base_path)) < self.train_count:
            raise AssertionError('Not all images were generated')

        path = self.all_scenes_path + '/all_scenes.json'
        with open(path, 'r') as f:
            all_scenes = json.load(f)
            for scene in all_scenes['scenes'][:train_count]:
                self.images.append(scene['image_filename'])
                # self.depths.append(scene['depth_map_filename'])
                train = jsonpickle.decode(scene['m_train'])
                self.trains.append(train)
                if 'mask' in (self.y_val + self.X_val) or 'position' in (self.y_val + self.X_val):
                    self.masks.append(scene['car_masks'])

        if self.X_val == 'gt_single_mask' or self.X_val == 'gt_positions':
            self.split_masks()

        if 'pred' in self.X_val or 'gt_single_mask' in self.X_val:
            from models.detectron import setup, predict_instances
            if not os.path.isfile(self.predictions_path):
                conf_path = "./configs/mask_rcnn_R_101_FPN_3x.yaml"
                cfg = setup(conf_path, self.base_scene, ds_typ)
                print('predicting instances using detectron')
                predict_instances(self.base_scene, ds_typ, cfg)

            with open(self.predictions_path, 'r') as f:
                predictions = json.load(f)
                self.predictions = [{}] * self.train_count
                for key, value in predictions.items():
                    file_name = value['file_name'].split("/")[-1]
                    ind = self.images.index(file_name)
                    self.predictions[ind] = value['instances']

        if 'pred_positions' in self.X_val:
            self.pred_world_cords = np.load(self.predictions_coords_path, allow_pickle=True)

        if 'predicted_attributes' in self.X_val:
            model_name = ['attr_predictor', 'resnet18'][1]
            # predictions_descriptions_path =
            # f'output/models/attr_predictor/attribute_classification/{self.train_col}/{self.base_scene}/predicted_descriptions/{train_count}/fold_0.npy'
            predicted_train_description = {
                100: np.load(f'output/models/{model_name}/attribute_classification/{ds_typ}/predicted_descriptions/100/fold_0.npy', allow_pickle=True),
                1000: np.load(f'output/models/{model_name}/attribute_classification/{ds_typ}/predicted_descriptions/1000/fold_0.npy', allow_pickle=True),
                8000: np.load(f'output/models/{model_name}/attribute_classification/{ds_typ}/predicted_descriptions/8000/fold_0.npy', allow_pickle=True)
            }
            attr = np.empty(0)
            for i in range(10000):
                attr_gt = self.get_attributes(i).numpy()
                attr = np.hstack((attr, attr_gt))

            acc8000 = accuracy_score(attr, predicted_train_description[8000].flatten())
            acc1000 = accuracy_score(attr, predicted_train_description[1000].flatten())
            acc100 = accuracy_score(attr, predicted_train_description[100].flatten())

            print(f'percent of correct predicted attributes: 100 ims: {acc100}, 1000 ims: {acc1000}, 8000 ims: {acc8000}')

            self.predicted_train_description = predicted_train_description

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
        # Return labels corresponding to class mode
        # depending on to segmentation mode:
        # returns image as item
        # returns segmentation as item
        if self.X_val == 'predicted_attributes':
            # train with predicted attributes from detectron
            X = self.get_pred_att(item)
        elif self.X_val == 'predicted_attributes_permutation_invariant':
            X = self.get_pred_att_per_inv(item)
        elif self.X_val == 'gt_attributes':
            # trains with ground truth attributes
            X = self.get_attributes(item).unsqueeze(dim=0).float()
        elif self.X_val == 'gt_attributes_individual_class':
            # trains with ground truth attributes where each attribute has its individual classes
            X = self.get_attributes_old(item).unsqueeze(dim=0).float()
        elif self.X_val == 'predicted_mask':
            # trains with predicted masks from detectron
            pred_masks, pred_classes = self.get_prediction(item)
            # print('hallo')
            # print(pred_masks.size())
            pred_classes_normed = pred_classes / len(self.attribute_classes)
            pred_masks = torch.moveaxis(pred_masks, 0, -1) * pred_classes_normed
            X = torch.moveaxis(pred_masks, -1, 0)
        elif self.X_val == 'gt_mask':
            # trains with ground truth maks
            mask = self.get_gt_masks(item)
            X = mask / len(self.attribute_classes)
        elif self.X_val == 'gt_single_mask':
            # trains with single ground truth mak concatenated with the original image
            X = self.gt_single_mask(item)

        elif self.X_val == 'gt_positions':
            labels = self.get_attributes(item)
            positions = self.get_gt_positions(item)
            X = torch.hstack([labels.unsqueeze(dim=1), positions])
            X_rand = torch.zeros_like(X)
            indices = torch.argsort(torch.rand_like(positions[:, 0]), dim=0)
            for i in range(32):
                X_rand[i] = X[indices[i]]
            X = X_rand
            # indices = torch.cat([indices.unsqueeze(dim=1), indices.unsqueeze(dim=1),indices.unsqueeze(dim=1)], dim=1)
            # result = torch.gather(positions, dim=0, index=indices)
        elif self.X_val == 'pred_positions':
            _, pred_classes = self.get_prediction(item)
            labels = pred_classes / len(self.attribute_classes)
            positions = self.get_pred_positions(item)
            X = torch.hstack([labels.unsqueeze(dim=1), positions])
        elif self.X_val == 'image':
            image = self.get_pil_image(item)
            X = self.norm(image)
        else:
            raise AssertionError(f'no valid X value specified for dataset: {self.X_val}')

        if self.y_val == 'direction':
            y = self.get_direction(item)
        elif self.y_val == 'attribute':
            y = self.get_attributes(item)
        elif self.y_val == 'attribute_str':
            att = self.get_attributes(item)
            y = [self.attribute_classes[val] for val in att]
        elif self.y_val == 'attribute_binary':
            att = self.get_attributes(item)
            y = torch.zeros(len(att) * len(self.attribute_classes))
            for c, val in enumerate(att):
                y[c * len(self.attribute_classes) + val] = 1
        elif self.y_val == 'attribute_individual_class':
            y = self.get_attributes_old(item)
        elif self.y_val == 'mask':
            y = self.get_gt_masks(item)
        elif self.y_val == 'gt_position':
            y = self.get_gt_position(item)
        else:
            raise AssertionError('no valid y label specified for dataset')
        return X, y

    def __len__(self):
        if self.X_val == 'gt_single_mask':
            return len(self.masks)
        else:
            return self.train_count

    def get_attributes(self, item):
        att = self.attribute_classes
        train = self.trains[item]
        cars = train.get_cars()
        labels = [0] * 32
        # each train has (4 cars a 8 attributes) totalling to 32 labels
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
            labels[8 * (car_number - 1):8 * car_number] = [color, length, wall, roof, wheels] + l_shapes
        return torch.tensor(labels)

    def get_attributes_old(self, item):
        # each label has its own classes
        load_shapes = ['rectangle', 'triangle', 'circle', 'diamond', 'hexagon', 'utriangle']
        lengths = ['short', 'long']
        roofs = ['none', 'arc', 'flat', 'jagged', 'peaked']
        shapes = ['rectangle', 'bucket', 'ellipse', 'hexagon', 'u_shaped']
        walls = ["double", "not_double"]
        wheel_count = ['2_wheels', '3_wheels']

        train = self.trains[item]
        cars = train.get_cars()
        labels = [0] * 28
        # shape output, 5 different shape + absence of car represented as index 0
        # length output, 2 different shape + absence of car represented as index 0
        # wall output, 2 different walls + absence of car represented as index 0
        # roof output, 4 different roof shapes + absence of car represented as index 0
        # wheels output, 2 different wheel counts + absence of car represented as index 0
        # load number output, max 3 payloads min 0
        # load shape output, 6 different shape + absence of car represented as index 0
        for car in cars:
            # index 0 = not existent
            car_number = car.get_car_number()
            shape = shapes.index(car.get_car_shape()) + 1
            length = lengths.index(car.get_car_length()) + 1
            double = walls.index(car.get_car_wall()) + 1
            roof = roofs.index(car.get_car_roof())
            wheels = wheel_count.index(car.get_car_wheels()) + 1
            l_num = car.get_load_number()
            l_shape = load_shapes.index(car.get_load_shape()) + 1
            labels[7 * (car_number - 1):7 * car_number] = [shape, length, double, roof, wheels, l_num, l_shape]
        return torch.tensor(labels)

    # def get_pred_att(self, item):
    #     """
    #     walls = ["braced_wall", 'solid_wall']
    #     load_obj = ['diamond', "box", "golden_vase", 'barrel', 'metal_pot', 'oval_vase']
    #     roofs = ["roof_foundation", 'solid_roof', 'braced_roof', 'peaked_roof']
    #     wheel_count = ['2_wheels', '3_wheels']
    #     color = ['yellow', 'green', 'grey', 'red', 'blue']
    #     length = ['short', 'long']
    #
    #
    #     returns predicted train attributes
    #     max number of detected objects
    #     - max 4 cars
    #         - max 3 load objects
    #         - car color
    #         - roof
    #         - wheel_count
    #         - length
    #         - wall
    #     max number = 4(3+1+1+1+1+1) = 32
    #     reserve 0 for not existent objects -> add 1 to attribute class index
    #     """
    #     attributes = torch.zeros(32)
    #     # seg_preds[0]['instances'][0].pred_boxes -> predicted boxes
    #     # seg_preds[0]['instances'][0].scores -> predicted class scores
    #     # seg_preds[0]['instances'][0].pred_classes -> predicted classes
    #     # seg_preds[0]['instances'][0].pred_masks -> predicted masks
    #     # im_name = self.predictions[str(item)]['file_name'].split("/")[-1]
    #     # ds_im_name = self.images[item]
    #
    #     instances = self.predictions[item]
    #     scores = np.asarray([value['score'] for id, value in instances.items()])
    #     idx_h = (-scores).argsort()[:32]
    #     for c, idx in enumerate(idx_h):
    #         if instances[str(idx)]['score'] > 0.9:
    #
    #             pred_class = int(instances[str(idx)]['pred_class']) + 1
    #             attributes[c] = pred_class
    #     # add 1 to attribute class index
    #     # for id, value in instances.items():
    #     #     attributes[int(id)] = int(value['pred_class'] + 1)
    #
    #     attributes = attributes.unsqueeze(dim=0)
    #     return attributes

    # def get_pred_masks(self, item):

    #
    #     instances = self.predictions[item]
    #     scores = np.asarray([value['score'] for id, value in instances.items()])
    #     idx_h = (-scores).argsort()[:32]
    #     # select a maximum of 32 predicted instances per image where score > 0.9
    #     masks = torch.zeros(32, 270, 480)
    #     for idx in idx_h:
    #         instance = instances[str(idx)]
    #         if instance['score'] > 0.9:
    #             masks[int(idx), :, :] = torch.from_numpy(
    #                 maskUtils.decode(instance['pred_mask']) * (instance['pred_class'] + 1))
    #
    #     return masks

    def get_gt_masks(self, item):
        masks = torch.zeros(32, 270, 480)
        mask = self.get_mask(item)
        attr_id = -1
        y = self.get_attributes(item)
        for car_id, car in mask.items():
            whole_car_mask = car['mask']
            whole_car_bbox = maskUtils.toBbox(whole_car_mask)
            del car['mask'], car['b_box'], car['world_cord']
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
                        class_id = self.attribute_classes.index(label)
                        masks[attr_id, :, :] = torch.from_numpy(maskUtils.decode(rle) * class_id)
                        if att_name == 'payload_1' or att_name == 'payload_2' or att_name == 'payload_0':
                            if masks[attr_id, :, :].max() != y[attr_id]:
                                raise AssertionError(
                                    f'class {label} with id {y[attr_id]}, rle id is {torch.from_numpy(maskUtils.decode(rle) * class_id).max()}')
                        #     print(f'class {label} with id {y[attr_id]}, rle id is {torch.from_numpy(maskUtils.decode(rle) * class_id).max()}')
                    del car[att_name]
                else:
                    if y[attr_id] != 0:
                        raise AssertionError(att_name + ' not in car')
        return masks

    def get_prediction(self, item):
        # seg_preds[0]['instances'][0].pred_boxes -> predicted boxes
        # seg_preds[0]['instances'][0].scores -> predicted class scores
        # seg_preds[0]['instances'][0].pred_classes -> predicted classes
        # seg_preds[0]['instances'][0].pred_masks -> predicted masks
        instances = self.predictions[item]
        scores = np.asarray([value['score'] for id, value in instances.items()])
        idx_h = (-scores).argsort()[:32]
        # select a maximum of 32 predicted instances per image where score > 0.9
        pred_masks = torch.zeros(32, 270, 480)
        pred_att = torch.zeros(32)

        for idx in idx_h:
            instance = instances[str(idx)]
            if instance['score'] > 0.9:
                pred_class = int(instance['pred_class']) + 1
                pred_att[int(idx)] = pred_class
                pred_masks[int(idx), :, :] = torch.from_numpy(
                    maskUtils.decode(instance['pred_mask']))
        return pred_masks, pred_att

    def split_masks(self):
        masks = {}
        train_ids = []
        positions = torch.empty(0, 3)
        position_by_id = torch.zeros(len(self.masks), 32, 3)
        mask_id = 0
        for train_id, train_masks in enumerate(self.masks):
            attr_id = -1
            y = self.get_attributes(train_id)
            train_ids.append(train_id)
            for car_id, car in train_masks.items():
                whole_car_mask = car['mask']
                whole_car_pos = car['world_cord']

                for att_name in ['color', 'length', 'wall', 'roof', 'wheels', 'payload_0', 'payload_1', 'payload_2']:
                    attr_id += 1
                    if att_name in car:
                        att = car[att_name]
                        label = att['label']
                        if label != 'none':
                            if att_name == 'length' or att_name == 'color':
                                rle = whole_car_mask
                                car_pos = whole_car_pos
                            else:
                                rle = att['mask']
                                car_pos = att['world_cord']

                            class_id = self.attribute_classes.index(label)
                            # mask = torch.from_numpy(maskUtils.decode(rle) * class_id).unsqueeze(dim=0)
                            # masks = torch.cat([masks, mask])
                            masks[mask_id] = {}
                            masks[mask_id]['class'] = class_id
                            masks[mask_id]['rle'] = rle
                            mask_id += 1
                            positions = torch.cat([positions, torch.tensor(car_pos).unsqueeze(dim=0)])
                            position_by_id[train_id, attr_id, :] = torch.tensor(car_pos)

                            # if att_name == 'payload_1' or att_name == 'payload_2' or att_name == 'payload_0':
                            #     if mask.max() != y[attr_id]:
                            #         raise AssertionError(
                            #             f'class {label} with id {y[attr_id]}, rle id is {torch.from_numpy(maskUtils.decode(rle) * class_id).max()}')
                    else:
                        if y[attr_id] != 0:
                            raise AssertionError(att_name + ' not in car')
        self.single_masks = masks
        # self.masks_ids = train_ids
        self.single_positions = positions
        # position_by_id -= torch.min(position_by_id)
        # position_by_id /= torch.max(position_by_id)
        self.position_normed = position_by_id

    def gt_single_mask(self, item):
        rle = self.single_masks[item]['rle']
        class_id = self.single_masks[item]['class']
        mask = torch.from_numpy(maskUtils.decode(rle) * class_id).unsqueeze(dim=0)
        mask = mask / len(self.attribute_classes)

        im_path = self.image_base_path + '/' + self.images[item]
        image = self.get_pil_image(item)
        X = self.norm(image)
        # apply mask to image directly
        # seg = image.clone()
        # for i in range(3):
        #     seg[i, :, :] = image[i, :, :] * mask.float()
        combined = torch.concat([X, mask])
        return combined

    def get_gt_positions(self, item):
        positions = torch.zeros(32, 3)
        labels = torch.zeros(32, dtype=torch.int8)
        mask = self.get_mask(item)
        attr_id = -1
        y = self.get_attributes(item)
        for car_id, car in mask.items():
            whole_car_pos = car['world_cord']
            # del car['mask'], car['b_box'], car['world_cord']
            for att_name in ['color', 'length', 'wall', 'roof', 'wheels', 'payload_0', 'payload_1', 'payload_2']:
                attr_id += 1
                if att_name in car:
                    att = car[att_name]
                    label = att['label']
                    if label != 'none':
                        if att_name == 'length' or att_name == 'color':
                            car_pos = whole_car_pos
                        else:
                            car_pos = att['world_cord']
                        class_id = self.attribute_classes.index(label)
                        positions[attr_id, :] = torch.tensor(car_pos)
                        labels[attr_id] = class_id
                        if att_name == 'payload_1' or att_name == 'payload_2' or att_name == 'payload_0':
                            if class_id != y[attr_id]:
                                raise AssertionError(f'class {label} with id {y[attr_id]}')
                    # del car[att_name]
                else:
                    if y[attr_id] != 0:
                        raise AssertionError(att_name + ' not in car')

        return positions
        # return self.position_normed[item]

    def get_pred_positions(self, item):
        predicted_position = torch.tensor(self.pred_world_cords[item], dtype=torch.float32)
        return predicted_position

    def get_pred_att(self, item):
        pred_att = torch.tensor(self.predicted_train_description[self.predictions_im_count][item], dtype=torch.float32)
        return pred_att.unsqueeze(dim=0)

    def get_pred_att_per_inv(self, item):
        pred_att = torch.tensor(self.predicted_train_description[self.predictions_im_count][item], dtype=torch.float32)
        car_pos = torch.zeros_like(pred_att)
        car_pos[8:] += 22
        car_pos[16:] += 22
        car_pos[24:] += 22
        pred_att += car_pos
        # pred_att = pred_att[pred_att != 0]
        # pred_att = pred_att[pred_att != 22]
        # pred_att = pred_att[pred_att != 44]
        # pred_att = pred_att[pred_att != 66]
        return pred_att.unsqueeze(dim=0)

    def get_gt_position(self, item):
        return self.single_positions[item]

    def get_direction(self, item):
        lab = self.trains[item].get_label()
        if lab == 'none':
            # return torch.tensor(0).unsqueeze(dim=0)
            raise AssertionError(f'There is no direction label for a RandomTrains. Use MichalskiTrain DS.')
        label_binary = self.direction_classes.index(lab)
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


class MichalskiTrainParser(Parser):
    def __init__(self, michalski_train_ds):
        super().__init__()
        self.michalski_train_ds = michalski_train_ds.dataset
        label_classes = self.michalski_train_ds.label_classes
        self.class_to_idx = {c: idx for idx, c in enumerate(label_classes)}

    def __getitem__(self, index):
        path = self.michalski_train_ds.get_image_path(index)
        target = self.michalski_train_ds.get_label_for_id()
        return open(path, "rb"), target

    def __len__(self):
        return self.michalski_train_ds.__len__()

    def _filename(self, item, basename=False, absolute=False):
        filename = self.michalski_train_ds.get_image_path(item)
        if basename:
            filename = os.path.basename(filename)
        elif not absolute:
            filename = os.path.relpath(filename, self.michalski_train_ds.image_base_path)
        return filename


def get_datasets(base_scene, raw_trains, train_vis, image_count, y_val='direction', X_val='image', total_image_count=10000,
                 resize=False, ):
    path_ori = f'output/image_generator/{raw_trains}/{train_vis}/{base_scene}'
    path_des = f'dataset/{raw_trains}/{train_vis}/{base_scene}'
    if not os.path.isdir(path_des):
        im_path = path_ori + '/images'
        if not os.path.isdir(im_path):
            raise AssertionError('dataset not found, please generate images first')

        files = os.listdir(im_path)
        # total image count equals 10.000 adjust if not all images need to be generated
        if len(files) != total_image_count:
            raise AssertionError(
                f'not enough images in dataset: expected {total_image_count}, present: {len(files)}'
                f' please generate the missing images')

        copy_tree(path_ori, path_des)
        shutil.rmtree(path_ori)
        if len(os.listdir(f'output/image_generator/{raw_trains}/{train_vis}/')) == 0:
            os.rmdir(f'output/image_generator/{raw_trains}/{train_vis}/')
    # merge json files to one if it does not already exist
    if not os.path.isfile(path_des + '/all_scenes/all_scenes.json'):
        merge_json_files(path_des)
    # image_count = None for standard image count
    full_ds = MichalskiTrainDataset(base_scene=base_scene, raw_trains=raw_trains, train_vis=train_vis, y_val=y_val,
                                    train_count=image_count, resize=resize, X_val=X_val)
    return full_ds


