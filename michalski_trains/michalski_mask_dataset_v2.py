import numpy as np
import torch
from pycocotools import mask as maskUtils

from michalski_trains.michalski_attribute_dataset import MichalskiAttributeDataset


class MichalskiMaskDatasetV2(MichalskiAttributeDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attribute_classes = self.blender_categories()

    def __getitem__(self, item):
        target = {}
        image = self.get_pil_image(item)
        X = self.norm(image)
        boxes = self.get_bboxes(item)
        labels = self.get_mask_labels(item)
        masks = self.get_masks(item)

        target['boxes'] = boxes
        target['labels'] = labels
        target['image_id'] = torch.tensor([item], dtype=torch.int64)
        target['area'] = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        target['iscrowd'] = torch.zeros_like(target['area'], dtype=torch.uint8)
        # not sure how crowd is to be defined in this dataset
        # target['iscrowd'] = torch.zeros((boxes.shape[0],), dtype=torch.int64)
        target['masks'] = masks
        if boxes.size()[0] != labels.size()[0] or boxes.size()[0] != masks.size()[0]:
            raise ValueError(
                f'gt size missmatch, boxes: {boxes.size()}, labels: {labels.size()}, masks: {masks.size()}')
        return X, target

    def blender_categories(self):
        ''' Returns a list of all possible categories in the dataset
        :return: list of categories [str] of length 24
        '''
        color = ['yellow', 'green', 'grey', 'red', 'blue']
        length = ['short', 'long']
        walls = ["braced_wall", 'solid_wall']
        roofs = ["roof_foundation", 'solid_roof', 'braced_roof', 'peaked_roof']
        wheel_count = ['2_wheels', '3_wheels']
        load_obj = ["box", "golden_vase", 'barrel', 'diamond', 'metal_pot', 'oval_vase']
        return ['none'] + color + length + walls + roofs + wheel_count + load_obj + ['car'] + ['locomotive']

    def get_mask_labels(self, item):
        ''' Returns a tensor of labels for all masks of the train.
        :param item: index of the train
        :return: tensor of labels
        '''
        att = self.attribute_classes
        train = self.trains[item]
        cars = train.get_cars()
        labels = [att.index('locomotive')]
        # each train has (n cars a 9 attributes) totalling to n*9 labels + 1 for the locomotive
        # each label can have 22 classes (assignable attributes) + 1 car label + 1 for the locomotive
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
            car_label = att.index('car')
            labels += [car_label, color, length, wall, roof, wheels] + l_shapes
        # remove the 0 labels (not existent)
        labels = torch.tensor(labels, dtype=torch.int64)
        return labels[labels != 0]

    def get_masks(self, item):
        ''' returns a list of masks of the train. Each mask is a tensor of shape (270, 480).
         It returns 1 mask of the locomotive and a maximum of 9 masks for each car in the train.
         Masks for not existent attributes are removed, e.g. if a car has no payload, the mask for payload_0 is removed.
         The order of the corresponding labels are:
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
            tensor of masks of shape (9*car numbers + 1, 270, 480), dtype = torch.uint8
        '''
        w, h = self.image_size
        masks = torch.empty(0, h, w, dtype=torch.uint8)
        mask = self.get_mask(item)
        attr_id = -1
        y = self.get_attributes(item)
        loco_mask = mask['loco']['mask']
        masks = torch.vstack([masks, torch.from_numpy(maskUtils.decode(loco_mask)).unsqueeze(0)])

        for car_id, car in mask.items():
            if 'car' not in car_id:
                continue
            whole_car_mask = car['mask']
            masks = torch.vstack([masks, torch.from_numpy(maskUtils.decode(whole_car_mask)).unsqueeze(0)])
            for att_name in ['color', 'length', 'wall', 'roof', 'wheels', 'payload_0', 'payload_1', 'payload_2']:
                attr_id += 1
                if att_name in car:
                    att = car[att_name]
                    label = att['label']
                    if label != 'none':
                        if att_name == 'length' or att_name == 'color' or (
                                att_name == 'roof' and self.train_vis == 'SimpleObjects'):
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

    def get_bboxes(self, item, format='[x0,y0,x1,y1]'):
        ''' returns a list of bounding boxes of the train. Each bounding box is a tensor of shape (270, 480).
        It returns 1 mask of the locomotive and 9 masks for each car in the train. The order of the corresponding labels are:
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
            FloatTensor of bounding boxes of shape (9*car numbers + 1, 270, 480)
        '''
        bboxes = torch.empty(0, 4)
        mask = self.get_mask(item)
        attr_id = -1
        y = self.get_attributes(item)
        # loco_bbox = mask['loco']['b_box']
        loco_bbox = maskUtils.toBbox(mask['loco']['mask'])
        loco_bbox = (loco_bbox + np.concatenate(([0, 0], loco_bbox[:2]))) if format == '[x0,y0,x1,y1]' else loco_bbox
        bboxes = torch.vstack([bboxes, torch.tensor(loco_bbox)])

        for car_id, car in mask.items():
            if 'car' not in car_id:
                continue
            whole_car_mask = car['mask']
            whole_car_bbox = maskUtils.toBbox(whole_car_mask)
            # whole_car_bbox = car['b_box']
            box_formated = (whole_car_bbox + np.concatenate(
                ([0, 0], whole_car_bbox[:2]))) if format == '[x0,y0,x1,y1]' else whole_car_bbox
            bboxes = torch.vstack([bboxes, torch.tensor(box_formated)])
            for att_name in ['color', 'length', 'wall', 'roof', 'wheels', 'payload_0', 'payload_1', 'payload_2']:
                attr_id += 1
                if att_name in car:
                    att = car[att_name]
                    label = att['label']
                    if label != 'none':
                        if att_name == 'length' or att_name == 'color' or (
                                att_name == 'roof' and self.train_vis == 'SimpleObjects'):
                            box = whole_car_bbox
                        else:
                            try:
                                # box = att['b_box']
                                box = maskUtils.toBbox(att['mask'])
                            except:
                                raise ValueError(
                                    f'b_box for {att} not found, for car {car_id} in train {item} in image at {self.get_image_path(item)}')
                        box_formated = (box + np.concatenate(([0, 0], box[:2]))) if format == '[x0,y0,x1,y1]' else box
                        bboxes = torch.vstack([bboxes, torch.tensor(box_formated)])
                        if y[attr_id] == 0:
                            raise AssertionError(
                                att_name + f' mask and label inconsistency in car {car_id} of train {item}')
                        if box_formated.tolist() == [0, 0, 0, 0]:
                            raise AssertionError(
                                f'empy bbox for attribute {att_name} with value {label} in car {car_id}'
                                f' of train {item} with image at {self.get_image_path(item)}')
                    else:
                        # bboxes = torch.vstack([bboxes, torch.zeros(1, 4)])
                        if y[attr_id] != 0:
                            raise AssertionError(
                                att_name + f' mask and label inconsistency in car {car_id} of train {item}')
                else:
                    # if object is not in image, add a zero bbox
                    # bboxes = torch.vstack([bboxes, torch.zeros(1, 4)])
                    if y[attr_id] != 0:
                        raise AssertionError(
                            att_name + f' mask and label inconsistency in car {car_id} of train {item}')
        # remove zero bboxes (not present in the image)
        # bboxes = bboxes[bboxes != 0].view(-1, 4)
        return bboxes
