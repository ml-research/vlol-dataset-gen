import os
import shutil
from pathlib import Path
import random
import numpy as np
import torch

from raw.gen_raw_trains import read_trains


def create_bk(ds_size=None, noise=0):
    train_raw = 'MichalskiTrains'
    train_vis = 'SimpleObjects'
    y_val = 'direction'
    X_val = 'gt_attributes'

    train_c = 0
    path = './ilp/popper/gt'
    path_2 = './ilp/popper/gt2'
    path_3 = './ilp/popper/gt3'
    path_dilp = './ilp/dilp/gt'
    path_aleph = './ilp/aleph/trains2'
    os.makedirs(path, exist_ok=True)
    os.makedirs(path_dilp, exist_ok=True)
    try:
        os.remove(path + '/bk.pl')
    except OSError:
        pass
    try:
        os.remove(path_2 + '/bk.pl')
    except OSError:
        pass
    try:
        os.remove(path_3 + '/bk.pl')
    except OSError:
        pass
    try:
        os.remove(path + '/exs.pl')
    except OSError:
        pass
    try:
        os.remove(path_aleph + '/train.n')
    except OSError:
        pass
    try:
        os.remove(path_aleph + '/train.f')
    except OSError:
        pass
    trains = read_trains(f'raw/datasets/{train_raw}.txt', toSimpleObjs=train_vis == 'SimpleObjects')
    trains = random.sample(trains, ds_size)
    with open(path + '/exs.pl', 'w+') as exs_file, open(path + '/bk.pl', 'w+') as bk_file, \
            open(path_2 + '/bk.pl', 'w+') as bk2_file, open(path_3 + '/bk.pl', 'w+') as bk3_file, open(
        path_dilp + '/positive.dilp', 'w+') as pos, open(path_dilp + '/negative.dilp', 'w+') as neg, open(
        path_aleph + '/train.f', 'w+') as aleph_pos, open(path_aleph + '/train.n', 'w+') as aleph_neg:
        ds_size = len(trains) if ds_size is None else ds_size
        if len(trains) < ds_size:
            raise AssertionError(f'not enough trains in DS {len(trains)} to create a bk of size {ds_size}')
        for train in trains[:ds_size]:
            ns = random.random()
            label = train.get_label()
            train_c += 1
            bk_file.write(f'train(t{train_c}).\n')
            bk2_file.write(f'train(t{train_c}).\n')
            bk3_file.write(f'train(t{train_c}).\n')
            label = 'pos' if label == 'east' else 'neg'
            # if train_c < 10:
            exs_file.write(f'{label}(eastbound(t{train_c})).\n')
            if label == 'pos':
                pos.write(f'target(t{train_c}).\n')
                aleph_pos.write(f'eastbound(t{train_c}).\n')
            else:
                neg.write(f'target(t{train_c}).\n')
                aleph_neg.write(f'eastbound(t{train_c}).\n')
            for car in train.get_cars():

                # add car to bk if car color is not none
                # car_label_names = np.array(ds.attribute_classes)[car.to(dtype=torch.int32).tolist()]
                # color, length, walls, roofs, wheel_count, load_obj1, load_obj2, load_obj3 = car_label_names
                if ns < noise:
                    car_number = car.get_car_number()

                    color = ['yellow', 'green', 'grey', 'red', 'blue'][np.random.randint(5)]
                    length = ['short', 'long'][np.random.randint(2)]
                    walls = ["braced_wall", 'solid_wall'][np.random.randint(2)]
                    roofs = ["roof_foundation", 'solid_roof', 'braced_roof', 'peaked_roof'][np.random.randint(4)]
                    wheel_count = ['2_wheels', '3_wheels'][np.random.randint(2)]
                    l_shape = ["box", "golden_vase", 'barrel', 'diamond', 'metal_pot', 'oval_vase'][
                        np.random.randint(6)]
                    l_num = np.random.randint(4)
                    load_obj1, load_obj2, load_obj3 = [l_shape] * l_num + ['none'] * (3 - l_num)
                else:
                    color = car.get_blender_car_color()
                    length = car.get_car_length()
                    walls = car.get_blender_wall()
                    roofs = car.get_blender_roof()
                    wheel_count = car.get_car_wheels()
                    l_shape = car.get_blender_payload()
                    l_num = car.get_load_number()
                    load_obj1, load_obj2, load_obj3 = [l_shape] * l_num + ['none'] * (3 - l_num)
                    car_number = car.get_car_number()

                bk_file.write(f'has_car(t{train_c},t{train_c}_c{car_number}).' + '\n')
                bk_file.write(f'car_num(t{train_c}_c{car_number},{car_number}).' + '\n')
                bk2_file.write(f'has_car(t{train_c},t{train_c}_c{car_number}).' + '\n')
                bk2_file.write(f'car_num(t{train_c}_c{car_number},{car_number}).' + '\n')
                bk3_file.write(f'has_car(t{train_c},t{train_c}_c{car_number}).' + '\n')
                bk3_file.write(f'car_num(t{train_c}_c{car_number},{car_number}).' + '\n')
                position = ['first', 'second', 'third', 'fourth']
                # bk_file.write(f'{position[car_number - 1]}_car(t{train_c}_c{car_number}).' + '\n')
                # # behind
                # for i in range(1, car_c):
                #     bk_file.write(f'behind(t{train_c}_c{car_c},t{train_c}_c{i}).' + '\n')
                # color
                bk_file.write(f'{color}(t{train_c}_c{car_number}).' + '\n')
                bk2_file.write(f'{color}(t{train_c}_c{car_number}_color).' + '\n')
                bk2_file.write(f'car_color(t{train_c}_c{car_number},t{train_c}_c{car_number}_color).' + '\n')
                bk3_file.write(f'car_color(t{train_c}_c{car_number},{color}).' + '\n')
                # length
                bk_file.write(f'{length}(t{train_c}_c{car_number}).' + '\n')
                bk2_file.write(f'{length}(t{train_c}_c{car_number}).' + '\n')
                bk3_file.write(f'{length}(t{train_c}_c{car_number}).' + '\n')
                # walls
                bk_file.write(f'{walls}(t{train_c}_c{car_number}).' + '\n')
                bk2_file.write(f'{walls}(t{train_c}_c{car_number}).' + '\n')
                bk3_file.write(f'{walls}(t{train_c}_c{car_number}).' + '\n')
                # roofs
                if roofs != 'none':
                    #     bk_file.write(f'roof_closed(t{train_c}_c{car_number}).' + '\n')
                    bk_file.write(f'{roofs}(t{train_c}_c{car_number}).' + '\n')
                    bk2_file.write(f'{roofs}(t{train_c}_c{car_number}_roof).' + '\n')
                else:
                    bk_file.write(f'roof_open(t{train_c}_c{car_number}).' + '\n')
                    bk2_file.write(f'roof_open(t{train_c}_c{car_number}_roof).' + '\n')

                bk2_file.write(f'has_roof2(t{train_c}_c{car_number},t{train_c}_c{car_number}_roof).' + '\n')
                bk3_file.write(f'has_roof2(t{train_c}_c{car_number},{roofs}).' + '\n')

                # wheel_count
                wheel_num = ['two', 'three'][int(wheel_count[0]) - 2]
                # bk_file.write(f'{wheel_num}{wheel_count[1:]}(t{train_c}_c{car_number}).' + '\n')
                bk_file.write(f'has_wheel0(t{train_c}_c{car_number},{wheel_count[0]}).' + '\n')
                bk2_file.write(f'has_wheel0(t{train_c}_c{car_number},{wheel_count[0]}).' + '\n')
                bk3_file.write(f'has_wheel0(t{train_c}_c{car_number},{wheel_count[0]}).' + '\n')

                # payload
                payload_num = 3 - [load_obj1, load_obj2, load_obj3].count('none')
                payload_n = ['zero', 'one', 'two', 'three'][payload_num]
                # bk_file.write(f'{payload_n}_load(t{train_c}_c{car_number}).\n')
                bk_file.write(f'load_num(t{train_c}_c{car_number},{l_num}).\n')
                bk2_file.write(f'load_num(t{train_c}_c{car_number},{l_num}).\n')
                bk3_file.write(f'load_num(t{train_c}_c{car_number},{l_num}).\n')

                if l_num > 0:
                    bk2_file.write(f'{l_shape}(t{train_c}_c{car_number}_payload).\n')
                    bk2_file.write(f'has_payload(t{train_c}_c{car_number},t{train_c}_c{car_number}_payload).\n')
                bk3_file.write(f'has_payload(t{train_c}_c{car_number},{l_shape}).\n')
                for p_c, payload in enumerate([load_obj1, load_obj2, load_obj3]):
                    if payload != 'none':
                        bk_file.write(f'{payload}(t{train_c}_c{car_number}_l{p_c}).\n')
                        bk_file.write(f'has_payload(t{train_c}_c{car_number},t{train_c}_c{car_number}_l{p_c}).\n')

    file = Path(path + '/bk.pl')
    file.write_text(
        "\n".join(
            sorted(
                file.read_text().split("\n")[:-1]
            )
        )
    )
    file = Path(path_2 + '/bk.pl')
    file.write_text(
        "\n".join(
            sorted(
                file.read_text().split("\n")[:-1]
            )
        )
    )
    file = Path(path_3 + '/bk.pl')
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
    try:
        os.remove(path_aleph + '/train.b')
    except OSError:
        pass
    with open(path_aleph + '/bias2', 'r') as bias, open(path_3 + '/bk.pl', 'r') as bk, open(path_aleph + '/train.b',
                                                                                            'w+') as comb:
        comb.write(bias.read() + '\n')
        comb.write(bk.read())

    shutil.copy(path + '/bk.pl', path_dilp + '/facts.dilp')
    shutil.copy(path + '/exs.pl', path_2 + '/exs.pl')
    shutil.copy(path + '/exs.pl', path_3 + '/exs.pl')
