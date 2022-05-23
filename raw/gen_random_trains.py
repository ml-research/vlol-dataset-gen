import os
import random


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
                car = str(n) + ' ' + shape + ' ' + length + ' ' + double + ' ' + roof + ' ' + wheels + ' ' + str(l_num) + ' ' + l_shape
                if j != 0:
                    car = ' ' + car
                m_cars = m_cars + car
                # m_cars.append(michalski.MichalskiCar(n, shape, length, double, roof, wheels, l_num, l_shape))
            # m_trains.append(michalski.MichalskiTrain(m_cars, None))
            text_file.write(m_cars + '\n')
