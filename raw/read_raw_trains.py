# if angle = None (default) then angle is selected at random else all trains have given angle
from random import randint

import m_train


def read_trains(file, with_occlusion=False, angle=None):
    m_trains = []
    f = open(file, "r")
    for line in f:
        m_cars = []
        l = line.split(' ')
        dir = l[0]
        for c in range(len(l) // 8):
            ind = c * 8
            # a = (l[ind+i] for i in range(8))
            m_cars.append(m_train.MichalskiCar(l[ind + 1], l[ind + 2], l[ind + 3], l[ind + 4], l[ind + 5], l[ind + 6],
                                               l[ind + 7], l[ind + 8].strip('\n')))
        t_angle = get_random_angle(with_occlusion, angle)
        m_trains.append(m_train.MichalskiTrain(m_cars, dir, t_angle))
    return m_trains


def get_random_angle(with_occlusion, angle=None):
    if angle is not None:
        train_dir = angle
    else:
        allowed_deg = [-60, 60] if with_occlusion else [-70, 70]
        train_dir = randint(allowed_deg[0], allowed_deg[1]) + 180 * randint(0, 1)
    return train_dir
