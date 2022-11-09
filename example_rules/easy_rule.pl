% easy rule:
%The train has a short and a long car with the same colour.

eastbound(Train):- has_car(Train,Car), has_car(Train,Car2), short(Car), long(Car2), car_color(Car, A), car_color(Car2, A).