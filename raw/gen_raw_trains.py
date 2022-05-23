import os

from raw.gen_michalski_trains import gen_raw_michalski_trains
from raw.gen_random_trains import gen_raw_random_trains


def gen_raw_trains(train_col):
    if not os.path.isfile(f'raw/{train_col}.txt'):
        if train_col == 'RandomTrains':
            gen_raw_random_trains()
        elif train_col == 'MichalskiTrains':
            gen_raw_michalski_trains()
