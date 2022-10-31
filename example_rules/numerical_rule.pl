% numerical rule:
eastbound(Train):- has_car(Train,Car), load_num(Car,N), car_num(Car,N), has_wheel0(Car,N).
% other numerical rules
% eastbound(Train):- has_car(Train,Car, N), load_num(Car, N), has_car(Train,Car2), has_wheel0(Car2, N), car_color(Car, A), car_color(Car2, A).
% eastbound(Train):- has_car(Train,Car, N), load_num(Car, N), has_car(Train,Car2, N2), has_wheel0(Car2, N), car_color(Car, A), car_color(Car2, A), N<N2.
