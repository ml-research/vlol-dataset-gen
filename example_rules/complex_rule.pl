% Complex rule:
% Either there is a car with a car number which is smaller as its wheel count and smaller as the number of loads,
% or there is a short and a long car with the same colour where the position number of the short car is smaller as the wheel count of the long car,
% or the train has three differently coloured cars.
eastbound(Train):- has_car(Train,Car), load_num(Car,N1), car_num(Car,N2), has_wheel0(Car,N3), N2 < N1, N2 < N3.
eastbound(Train):- has_car(Train,Car, N1), has_car(Train,Car2), short(Car), long(Car2), car_color(Car, A), car_color(Car2, A), has_wheel0(Car2,N2), N1 < N2.
eastbound(Train):- has_car(Train,B), has_car(Train,C), has_car(Train,D), car_color(D,X), car_color(C,Y), car_color(B,Z), X\=Y, Y\=Z, Z\=X.