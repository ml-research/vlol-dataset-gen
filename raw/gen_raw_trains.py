import math
import os
import random
import shutil
from pyswip import Prolog

from raw import read_raw_trains


def gen_raw_michalski_trains(num_entries=10000):
    with open("raw/train_generator.pl", 'r') as generator, open("classification_rule.pl", 'r') as rule:
        with open('raw/tmp/generator_tmp.pl', 'w+') as generator_tmp:
            generator_tmp.write(generator.read())
            generator_tmp.write(rule.read())
    generator_tmp = 'raw/tmp/generator_tmp.pl'

    prolog = Prolog()
    prolog.consult(generator_tmp)
    all_trains_p = "raw/datasets/MichalskiTrains.txt"
    try:
        os.remove(all_trains_p)
    except OSError:
        pass
    with open(all_trains_p, 'w+') as all_trains:
        west_counter, east_counter = 0, 0
        while west_counter < num_entries / 2 or east_counter < num_entries / 2:
            for res in prolog.query(f"loop({1})."):
                continue
            train = open('raw/tmp/MichalskiTrains.txt', 'r').read()
            if 'east' in train and east_counter < int(num_entries / 2):
                all_trains.write(train)
                east_counter += 1
            elif 'west' in train and west_counter < int(math.ceil(num_entries / 2)):
                all_trains.write(train)
                west_counter += 1
            os.remove('raw/tmp/MichalskiTrains.txt')
        print(f'generated {west_counter} westbound trains and {east_counter} eastbound trains')
    os.remove(generator_tmp)


def gen_raw_random_trains(num_entries=10000):
    try:
        os.remove('RandomTrains')
    except OSError:
        pass
    with open('raw/RandomTrains.txt', 'w+') as text_file:
        m_trains = []
        for i in range(num_entries):
            m_cars = 'none '

            num_cars = random.randint(3, 4)
            for j in range(num_cars):
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
                car = str(n) + ' ' + shape + ' ' + length + ' ' + double + ' ' + roof + ' ' + wheels + ' ' + str(
                    l_num) + ' ' + l_shape
                if j != 0:
                    car = ' ' + car
                m_cars = m_cars + car
                # m_cars.append(michalski.MichalskiCar(n, shape, length, double, roof, wheels, l_num, l_shape))
            # m_trains.append(michalski.MichalskiTrain(m_cars, None))
            text_file.write(m_cars + '\n')


def gen_raw_trains(train_col, num_entries=10000, replace_exisiting=True):
    os.makedirs("raw/datasets/", exist_ok=True)
    if replace_exisiting:
        if train_col == 'RandomTrains':
            gen_raw_random_trains(num_entries)
        elif train_col == 'MichalskiTrains':
            gen_raw_michalski_trains(num_entries)


def generate_m_train_attr_overview(file):
    double = []
    l_num = []
    l_shape = []
    length = []
    car_position = []
    roof = []
    shape = []
    wheels = []
    trains = read_raw_trains.read_trains(file)
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
