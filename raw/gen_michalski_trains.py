import shutil

from pyswip import Prolog


def gen_raw_michalski_trains(num_entries=10000):
    prolog = Prolog()
    prolog.consult("raw/train_generator.pl")
    for res in prolog.query(f"loop({num_entries})."):
        print(res)
    shutil.move("MichalskiTrains.txt", "raw/MichalskiTrains.txt")
