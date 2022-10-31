% complex rule
eastbound(Train):- has_car(Train,Car), load_num(Car,N1), car_num(Car,N2), has_wheel0(Car,N3), N2 < N1, N2 < N3.
eastbound(Train):- has_car(Train,Car, N1), has_car(Train,Car2), short(Car), long(Car2), car_color(Car, A), car_color(Car2, A), has_wheel0(Car2,N2), N1 < N2.
eastbound(Train):- has_car(Train,B), has_car(Train,C), has_car(Train,D), car_color(D,X), car_color(C,Y), car_color(B,Z), X\=Y, Y\=Z, Z\=X.