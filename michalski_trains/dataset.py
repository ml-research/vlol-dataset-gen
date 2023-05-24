import glob
import json
import os
import shutil
import warnings
from datetime import datetime

import torch

from michalski_trains.michalski_attribute_dataset import MichalskiAttributeDataset
from michalski_trains.michalski_dataset import MichalskiDataset
from michalski_trains.michalski_mask_dataset import MichalskiMaskDataset
from michalski_trains.michalski_mask_dataset_v2 import MichalskiMaskDatasetV2


def get_datasets(base_scene, raw_trains, train_vis, class_rule, min_car=2, max_car=4,
                 ds_size=12000, ds_path='output/image_generator', y_val='direction', resize=False,
                 preprocessing=None, fixed_output_car_size=4):
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
                                   ds_size=ds_size, resize=resize, ds_path=ds_path,
                                   preprocessing=preprocessing)
    elif y_val == 'attributes':
        full_ds = MichalskiAttributeDataset(class_rule=class_rule, base_scene=base_scene, raw_trains=raw_trains,
                                            train_vis=train_vis, min_car=min_car, max_car=max_car,
                                            ds_size=ds_size, resize=resize, ds_path=ds_path,
                                            preprocessing=preprocessing)
    elif y_val == 'mask':
        full_ds = MichalskiMaskDataset(class_rule=class_rule, base_scene=base_scene, raw_trains=raw_trains,
                                       train_vis=train_vis, min_car=min_car, max_car=max_car,
                                       ds_size=ds_size, resize=resize, ds_path=ds_path,
                                       preprocessing=preprocessing)
    elif y_val == 'maskv2':
        full_ds = MichalskiMaskDatasetV2(class_rule=class_rule, base_scene=base_scene, raw_trains=raw_trains,
                                         train_vis=train_vis, min_car=min_car, max_car=max_car,
                                         ds_size=ds_size, resize=resize, ds_path=ds_path,
                                         preprocessing=preprocessing)
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



def blender_categories():
    color = ['yellow', 'green', 'grey', 'red', 'blue']
    length = ['short', 'long']
    walls = ["braced_wall", 'solid_wall']
    roofs = ["roof_foundation", 'solid_roof', 'braced_roof', 'peaked_roof']
    wheel_count = ['2_wheels', '3_wheels']
    load_obj = ["box", "golden_vase", 'barrel', 'diamond', 'metal_pot', 'oval_vase']
    return ['none'] + color + length + walls + roofs + wheel_count + load_obj


def original_categories():
    shape = ['rectangle', 'bucket', 'ellipse', 'hexagon', 'u_shaped']
    length = ['short', 'long']
    walls = ["double", 'not_double']
    roofs = ['arc', 'flat', 'jagged', 'peaked']
    wheel_count = ['2', '3']
    load_obj = ["rectangle", "triangle", 'circle', 'diamond', 'hexagon', 'utriangle']
    return ['none'] + shape + length + walls + roofs + wheel_count + load_obj


def michalski_labels():
    return ['color', 'length', 'walls', 'roofs', 'wheel_count', 'load_obj1', 'load_obj2', 'load_obj3']


def rcnn_blender_categories():
    cat = blender_categories()
    # cat += [f'car_{i}' for i in range(1, 21)]
    cat += ['car', 'locomotive']
    return cat


def rcnn_michalski_labels():
    return ['car_number', 'color', 'length', 'walls', 'roofs', 'wheel_count', 'load_obj']
