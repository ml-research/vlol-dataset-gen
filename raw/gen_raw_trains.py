import math
import os
import random
from pyswip import Prolog

from michalski_trains.m_train import BlenderCar, MichalskiTrain, SimpleCar


def gen_raw_michalski_trains(class_rule, out_path, num_entries=10000, with_occlusion=False):
    """ Generate Michalski trains descriptions using the Prolog train generator
        labels are derived by the classification rule
    Args:
        out_path: string path to save the generated train descriptions
        num_entries: int, number of michalski trains which are generated
        with_occlusion: boolean, whether to include occlusion of the train payloads
        class_rule: str, classification rule used to derive the labels
    """
    rule_path = f'example_rules/{class_rule}_rule.pl'
    os.makedirs('raw/tmp/', exist_ok=True)
    generator_tmp = 'raw/tmp/generator_tmp.pl'
    try:
        os.remove(generator_tmp)
    except OSError:
        pass
    with open("raw/train_generator.pl", 'r') as gen, open(rule_path, 'r') as rule:
        with open(generator_tmp, 'w+') as generator:
            generator.write(gen.read())
            generator.write(rule.read())

    prolog = Prolog()
    prolog.consult(generator_tmp)
    try:
        os.remove(out_path)
    except OSError:
        pass
    with open(out_path, 'w+') as all_trains:
        west_counter, east_counter = 0, 0
        while west_counter < num_entries / 2 or east_counter < num_entries / 2:
            try:
                os.remove(f'raw/tmp/MichalskiTrains.txt')
            except OSError:
                pass
            for _ in prolog.query(f"loop({1})."):
                continue
            train = open('raw/tmp/MichalskiTrains.txt', 'r').read()
            t_angle = get_random_angle(with_occlusion)
            tmp = train.split(" ", 1)
            train = tmp[0] + f' {t_angle} ' + tmp[1]
            if 'east' in train and east_counter < int(num_entries / 2):
                all_trains.write(train)
                east_counter += 1
            elif 'west' in train and west_counter < int(math.ceil(num_entries / 2)):
                all_trains.write(train)
                west_counter += 1
            os.remove('raw/tmp/MichalskiTrains.txt')
        print(f'generated {west_counter} westbound trains and {east_counter} eastbound trains')
    os.remove(generator_tmp)


def gen_raw_random_trains(class_rule, out_path, num_entries=10000, with_occlusion=False):
    """ Generate random trains descriptions
    Args:
        out_path: string path to save the generated train descriptions
        class_rule: str, classification rule used to derive the labels
        num_entries: int number of michalski trains which are generated
        with_occlusion: boolean whether to include occlusion of the train payloads
    """
    classifier = 'raw/tmp/concept_tester_tmp.pl'
    os.makedirs('raw/tmp/', exist_ok=True)
    rule_path = f'example_rules/{class_rule}_rule.pl'

    try:
        os.remove(classifier)
    except OSError:
        pass
    with open("raw/train_generator.pl", 'r') as gen, open(rule_path, 'r') as rule:
        with open(classifier, 'w+') as generator:
            generator.write(gen.read())
            generator.write(rule.read())
    prolog = Prolog()
    prolog.consult(classifier)
    west_counter = 0
    east_counter = 0

    try:
        os.remove(out_path)
    except OSError:
        pass
    with open(out_path, 'w+') as text_file:
        while west_counter < num_entries / 2 or east_counter < num_entries / 2:
            t_angle = get_random_angle(with_occlusion)
            train = ''
            m_cars = f''

            num_cars = random.randint(3, 4)
            for j in range(num_cars):
                train += ', ' if len(train) > 0 else ''

                n = j + 1
                length = random.choice(['short', 'long'])

                if length == 'long':
                    wheels = random.choice(['2', '3'])
                    l_num = random.randint(0, 3)
                else:
                    wheels = '2'
                    l_num = random.randint(0, 2)

                shape = random.choice(['rectangle', 'bucket', 'ellipse', 'hexagon', 'u_shaped'])
                double = random.choice(['not_double', 'double'])
                roof = random.choice(['none', 'arc', 'flat', 'jagged', 'peaked'])
                l_shape = random.choice(['rectangle', 'triangle', 'circle', 'diamond', 'hexagon', 'utriangle'])
                car = str(
                    n) + ' ' + shape + ' ' + length + ' ' + double + ' ' + roof + ' ' + wheels + ' ' + l_shape + ' ' + str(
                    l_num)
                train += f'c({str(n)}, {shape}, {length}, {double}, {roof}, {wheels}, l({l_shape}, {str(l_num)}))'

                if j != 0:
                    car = ' ' + car
                m_cars = m_cars + car
                # m_cars.append(michalski.MichalskiCar(n, shape, length, double, roof, wheels, l_num, l_shape))
            # m_trains.append(michalski.MichalskiTrain(m_cars, None))
            q = list(prolog.query(f"eastbound([{train}])."))
            p = 'west' if len(q) == 0 else 'east'
            if p == 'east' and east_counter < int(num_entries / 2):
                m_cars = f'{p} {t_angle} ' + m_cars
                text_file.write(m_cars + '\n')
                east_counter += 1
            if p == 'west' and west_counter < int(math.ceil(num_entries / 2)):
                m_cars = f'{p} {t_angle} ' + m_cars
                text_file.write(m_cars + '\n')
                west_counter += 1
    print(f'generated {west_counter} westbound trains and {east_counter} eastbound trains')
    os.remove(classifier)


def gen_raw_trains(train_col, classification_rule, out_path, num_entries=10000, replace_existing=True, with_occlusion=False):
    """ Generate random or Michalski train descriptions
    Args:
        train_col: string type of train which is generated available options: 'RandomTrains' and 'MichalskiTrains'
        out_path: string path to save the generated train descriptions
        classification_rule: str, path to classification rule used to derive the labels
        num_entries: int number of michalski trains which are generated
        replace_existing: bool whether the existing copy shall be replaced by a new copy
        with_occlusion: boolean whether to include occlusion of the train payloads
    """
    os.makedirs("raw/datasets/", exist_ok=True)
    if replace_existing:
        if train_col == 'RandomTrains':
            gen_raw_random_trains(classification_rule, out_path, num_entries, with_occlusion)
        elif train_col == 'MichalskiTrains':
            gen_raw_michalski_trains(classification_rule, out_path, num_entries, with_occlusion)


def read_trains(file, toSimpleObjs=False):
    """ read the trains generated by the prolog train generator
    Args:
        file: int number of michalski trains which are generated
        toSimpleObjs: if train is transformed to simple objects we need to update the pass indices accordingly
    """
    m_trains = []
    f = open(file, "r")
    for line in f:
        m_cars = []
        l = line.split(' ')
        dir = l[0]
        t_angle = l[1]
        for c in range(len(l) // 8):
            ind = c * 8
            # a = (l[ind+i] for i in range(8))
            car = BlenderCar(l[ind + 2], l[ind + 3], l[ind + 4], l[ind + 5], l[ind + 6], l[ind + 7], l[ind + 8],
                             l[ind + 9].strip('\n'))
            if toSimpleObjs:
                car = SimpleCar(l[ind + 2], l[ind + 3], l[ind + 4], l[ind + 5], l[ind + 6], l[ind + 7], l[ind + 8],
                                l[ind + 9].strip('\n'))

            m_cars.append(car)
        train = MichalskiTrain(m_cars, dir, t_angle)
        if toSimpleObjs is True:
            train.update_pass_indices()
        # t_angle = get_random_angle(with_occlusion, angle)
        m_trains.append(train)
    return m_trains


def get_random_angle(with_occlusion, angle=None):
    """ randomly sample an angle of the train
    Args:
        with_occlusion: boolean whether to include occlusion of the train payloads
        angle: int fixed angle, None to sample a new angle value
    """
    if angle is not None:
        train_dir = angle
    else:
        allowed_deg = [-60, 60] if with_occlusion else [-70, 70]
        train_dir = random.randint(allowed_deg[0], allowed_deg[1]) + 180 * random.randint(0, 1)
    return train_dir


def generate_m_train_attr_overview(file):
    """ Generates an overview of the train descriptions used by the train generator
    Args:
        file: string path to trains which shall be analysed
    """
    double = []
    l_num = []
    l_shape = []
    length = []
    car_position = []
    roof = []
    shape = []
    wheels = []
    trains = read_trains(file)
    for train in trains:
        for car in train.m_cars:
            if car.double not in double:
                double.append(car.double)
            if car.l_num not in l_num:
                l_num.append(car.l_num)
            if car.l_shape not in l_shape:
                l_shape.append(car.l_shape)
            if car.length not in length:
                length.append(car.length)
            if car.n not in car_position:
                car_position.append(car.n)
            if car.roof not in roof:
                roof.append(car.roof)
            if car.shape not in shape:
                shape.append(car.shape)
            if car.wheels not in wheels:
                wheels.append(car.wheels)

    with open("old/class_att", "w+") as text_file:
        text_file.write("double values:\n %s\n" % double)
        text_file.write("load numbers:\n %s\n" % l_num)
        text_file.write("load shapes:\n %s\n" % l_shape)
        text_file.write("length:\n %s\n" % length)
        text_file.write("car positions:\n %s\n" % car_position)
        text_file.write("roofs:\n %s\n" % roof)
        text_file.write("shapes:\n %s\n" % shape)
        text_file.write("wheels:\n %s\n" % wheels)
