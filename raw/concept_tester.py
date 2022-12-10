import os
import random
from importlib import reload

import pyswip
from pyswip import newModule, Prolog


def eval_rule(theory: str = None, ds_val: str = None, ds_train: str = None, dir='', print_stats=True, clean_up=True):
    name = f'tmp_{random.randint(1, 1000)}'
    # test1 = newModule(name)

    concept_tester_tmp = dir + f'tmp/concept_tester/{name}.pl'
    os.makedirs(dir + f'tmp/concept_tester', exist_ok=True)
    try:
        os.remove(concept_tester_tmp)
    except OSError:
        pass
    with open(f"{dir}raw/train_generator.pl", 'r') as gen, open(f"{dir}raw/concept_tester.pl", 'r') as rule:
        with open(concept_tester_tmp, 'w+') as generator:
            generator.write(':- dynamic eastbound/1 .')
            generator.write(gen.read())
            if theory is None:
                generator.write(rule.read())
            else:
                generator.write(theory)

    prolog = Prolog()
    if bool(list(prolog.query(f'source_file(\'{os.path.abspath(concept_tester_tmp)}\').'))):
        raise EnvironmentError('previous source file loaded')
    # prolog._init_prolog_thread()
    prolog.consult(os.path.abspath(concept_tester_tmp))
    # prolog.query(f'load_files([library(\'{os.path.abspath(concept_tester_tmp)})\']).')
    ds_val = f'output/image_generator/dataset_descriptions/MichalskiTrains_theoryx.txt' if ds_val is None else ds_val
    datasets = [ds_train, ds_val]
    TP, FN, TN, FP, TP_train, FN_train, TN_train, FP_train = [0] * 8

    for ds_t, ds in enumerate(datasets):
        if ds is not None:
            with open(ds, "r") as f:
                for line in f:
                    l = line.split(' ')
                    l = l[:-1] + [l[-1][:-1]]
                    dir = l[0]
                    train = ''
                    for c in range(len(l) // 8):
                        ind = c * 8
                        train += ', ' if c > 0 else ''
                        train += f'c({l[ind + 2]}, {l[ind + 3]}, {l[ind + 4]}, {l[ind + 5]}, {l[ind + 6]}, {l[ind + 7]}, l({l[ind + 8]}, {l[ind + 9]}))'
                    q = list(prolog.query(f"eastbound([{train}])."))
                    p = 'east' if bool(q) else 'west'

                    if p == 'east' and dir == 'east':
                        if ds_t == 0:
                            TP_train += 1
                        if ds_t == 1:
                            TP += 1
                    elif p == 'east' and dir == 'west':
                        if ds_t == 0:
                            FP_train += 1
                        if ds_t == 1:
                            FP += 1
                    elif p == 'west' and dir == 'west':
                        if ds_t == 0:
                            TN_train += 1
                        if ds_t == 1:
                            TN += 1
                    elif p == 'west' and dir == 'east':
                        if ds_t == 0:
                            FN_train += 1
                        if ds_t == 1:
                            FN += 1
                    # if p != dir:
                    # print(dir + str(q))
                    # print(train)

    if print_stats:
        if ds_train is not None:
            print(f'training: ACC:{(TP_train + TN_train) / (FN_train + TN_train + TP_train + FP_train)}, '
                  f'Precision:{(TP_train / (TP_train + FP_train)) if (TP_train + FP_train) > 0 else 0}, Recall:{(TP_train / (TP_train + FN_train)) if (TP_train + FN_train) > 0 else 0}, '
                  f'TP:{TP_train}, FN:{FN_train}, TN:{TN_train}, FP:{FP_train}')
        if ds_val is not None:
            print(f'Validation: ACC:{(TP + TN) / (FN + TN + TP + FP)}, '
                  f'Precision:{(TP / (TP + FP)) if (TP + FP) > 0 else 0}, Recall:{(TP / (TP + FN)) if (TP + FN) > 0 else 0}, '
                  f'TP:{TP}, FN:{FN}, TN:{TN}, FP:{FP}')
    # for fact in theory.split('\n'):
    # a = prolog.dynamic('eastbound')
    # prolog.retractall(f'eastbound(A)')

    list(prolog.query(f'unload_file(\'{os.path.abspath(concept_tester_tmp)}\').'))
    # list(prolog.query(f'unload_file(\'{concept_tester_tmp}\').'))
    del prolog
    if clean_up:
        os.remove(concept_tester_tmp)
    return TP, FN, TN, FP, TP_train, FN_train, TN_train, FP_train
