def eval_rule():
    from pyswip import Prolog, Query, Functor
    train_descriptions = f'raw/datasets/MichalskiTrains.txt'
    concept_tester = 'raw/concept_tester.pl'
    prolog = Prolog()
    prolog.consult(concept_tester)
    dirs = ['west', 'east']
    f = open(train_descriptions, "r")
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    for line in f:
        l = line.split(' ')
        dir = l[0]
        train = ''
        for c in range(len(l) // 8):
            ind = c * 8
            train += ', ' if c > 0 else ''
            train += f'c({l[ind + 2]}, {l[ind + 3]}, {l[ind + 4]}, {l[ind + 5]}, {l[ind + 6]}, {l[ind + 7]}, l({l[ind + 8]}, {l[ind + 9]}))'
        q = list(prolog.query(f"eastbound([{train}])"))
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
    print(f'concept tester')
    print(
        f'ACC:{(TP + TN) / (FN + TN + TP + FP)}, Precision:{TP / (TP + FP)}, Recall:{TP / (TP + FN)}, TP:{TP}, FN:{FN}, TN:{TN}, FP:{FP}')
