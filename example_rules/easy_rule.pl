% easy rule:
train with a short and a long car which have the same color
eastbound(Train):- has_car(Train,Car), has_car(Train,Car2), short(Car), long(Car2), car_color(Car, A), car_color(Car2, A).