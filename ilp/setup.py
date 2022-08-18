import os
import shutil
from pathlib import Path

import numpy as np
import torch

from raw.gen_raw_trains import read_trains


def create_bk(base_scene, ds_size, noise=0.01):
    train_raw = 'MichalskiTrains'
    train_vis = 'SimpleObjects'
    y_val = 'direction'
    X_val = 'gt_attributes'

    train_c = 0
    path = './ilp/popper/gt/'
    path_dilp = './ilp/dilp/gt/'
    os.makedirs(path, exist_ok=True)
    os.makedirs(path_dilp, exist_ok=True)
    try:
        os.remove(path + '/bk.pl')
    except OSError:
        pass
    try:
        os.remove(path + '/exs.pl')
    except OSError:
        pass
    trains = read_trains(f'raw/datasets/{train_raw}.txt', toSimpleObjs=train_vis == 'SimpleObjects')
    with open(path + '/exs.pl', 'w+') as exs_file, open(path + '/bk.pl', 'w+') as bk_file, open(
            path_dilp + '/positive.dilp', 'w+') as pos, open(path_dilp + '/negative.dilp', 'w+') as neg:
        for train in trains:
            n = np.random.random()
            label = train.get_label()
            car_c = 0
            train_c += 1
            bk_file.write(f'train(t{train_c}).\n')
            label = 'pos' if label == 'east' else 'neg'
            # if train_c < 10:
            exs_file.write(f'{label}(f(t{train_c})).\n')
            if label == 'pos':
                pos.write(f'target(t{train_c}).\n')
            else:
                neg.write(f'target(t{train_c}).\n')
            for car in train.get_cars():

                # add car to bk if car color is not none
                # car_label_names = np.array(ds.attribute_classes)[car.to(dtype=torch.int32).tolist()]
                # color, length, walls, roofs, wheel_count, load_obj1, load_obj2, load_obj3 = car_label_names

                color = car.get_blender_car_color()
                length = car.get_car_length()
                walls = car.get_blender_wall()
                roofs = car.get_blender_roof()
                wheel_count = car.get_car_wheels()
                l_shape = car.get_blender_payload()
                l_num = car.get_load_number()
                load_obj1, load_obj2, load_obj3 = [l_shape] * l_num + ['none'] * (3 - l_num)
                car_number = car.get_car_number()

                car_c += 1
                bk_file.write(f'has_car(t{train_c},t{train_c}_c{car_c}).' + '\n')
                position = ['first', 'second', 'third', 'fourth']
                bk_file.write(f'{position[car_c-1]}_car(t{train_c}_c{car_c}).' + '\n')
                # # behind
                # for i in range(1, car_c):
                #     bk_file.write(f'behind(t{train_c}_c{car_c},t{train_c}_c{i}).' + '\n')
                # color
                bk_file.write(f'{color}(t{train_c}_c{car_c}).' + '\n')
                # length
                bk_file.write(f'{length}(t{train_c}_c{car_c}).' + '\n')
                # walls
                bk_file.write(f'{walls}(t{train_c}_c{car_c}).' + '\n')
                # roofs
                if roofs != 'none':
                    bk_file.write(f'roof_closed(t{train_c}_c{car_c}).' + '\n')
                    bk_file.write(f'{roofs}(t{train_c}_c{car_c}).' + '\n')
                else:
                    bk_file.write(f'roof_open(t{train_c}_c{car_c}).' + '\n')
                # wheel_count
                wheel_num = ['two', 'three'][int(wheel_count[0]) - 2]
                bk_file.write(f'{wheel_num}{wheel_count[1:]}(t{train_c}_c{car_c}).' + '\n')
                # payload
                payload_num = 3 - [load_obj1, load_obj2, load_obj3].count('none')
                payload_n = ['zero', 'one', 'two', 'three'][payload_num]
                bk_file.write(f'{payload_n}_load(t{train_c}_c{car_c}).\n')
                for p_c, payload in enumerate([load_obj1, load_obj2, load_obj3]):
                    if payload != 'none':
                        bk_file.write(f'{payload}(t{train_c}_c{car_c}_l{p_c}).\n')
                        bk_file.write(f'has_load(t{train_c}_c{car_c},t{train_c}_c{car_c}_l{p_c}).\n')

    file = Path(path + '/bk.pl')
    file.write_text(
        "\n".join(
            sorted(
                file.read_text().split("\n")[:-1]
            )
        )
    )
    file = Path(path + '/exs.pl')
    file.write_text(
        "\n".join(
            sorted(
                file.read_text().split("\n")[:-1]
            )
        )
    )
    shutil.copy(path + '/bk.pl', path_dilp + '/facts.dilp')
