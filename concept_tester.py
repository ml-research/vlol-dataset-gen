import os


def eval_rule():
    from pyswip import Prolog, Query, Functor

    concept_tester_tmp = 'raw/tmp/concept_tester_tmp.pl'
    try:
        os.remove(concept_tester_tmp)
    except OSError:
        pass
    with open("raw/train_generator.pl", 'r') as gen, open("raw/concept_tester.pl", 'r') as rule:
        with open(concept_tester_tmp, 'w+') as generator:
            generator.write(gen.read())
            generator.write(rule.read())

    train_descriptions = f'raw/datasets/MichalskiTrains.txt'
    # concept_tester = 'raw/concept_tester.pl'
    prolog = Prolog()
    prolog.consult(concept_tester_tmp)
    dirs = ['west', 'east']
    f = open(train_descriptions, "r")
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    for line in f:
        l = line.split(' ')
        l = l[:-1] + [l[-1][:-1]]
        dir = l[0]
        train = ''
        for c in range(len(l) // 8):
            ind = c * 8
            train += ', ' if c > 0 else ''
            train += f'c({l[ind + 2]}, {l[ind + 3]}, {l[ind + 4]}, {l[ind + 5]}, {l[ind + 6]}, {l[ind + 7]}, l({l[ind + 8]}, {l[ind + 9]}))'
        print(train)
        q = list(prolog.query(f"eastbound([{train}])."))
        # q = list(prolog.query(f"loves(b, b)."))
        # print(q)
        p = 'west' if len(q) == 0 else 'east'
        if p == 'east' and dir == 'east':
            TP += 1
        elif p == 'east' and dir == 'west':
            FP += 1
        elif p == 'west' and dir == 'west':
            TN += 1
        elif p == 'west' and dir == 'east':
            FN += 1
        if p != dir:
            print(dir + str(q))
            print(train)
    print(f'concept tester')
    print(f'ACC:{(TP + TN) / (FN + TN + TP + FP)}, '
          f'Precision:{(TP / (TP + FP)) if (TP + FP) > 0 else 0}, Recall:{(TP / (TP + FN))if (TP + FN) > 0 else 0}, '
          f'TP:{TP}, FN:{FN}, TN:{TN}, FP:{FP}')
    os.remove(concept_tester_tmp)
