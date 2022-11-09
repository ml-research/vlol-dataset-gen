% multi rule:
% train with braced_wall and 2 loads, load_num 3 and a blue car or a blue car and braced_wall

eastbound(Train):- has_car(Train,Car), braced_wall(Car), has_car(Train,Car2), load_num(Car2, 2).
eastbound(Train):- has_car(Train,Car), load_num(Car,3), has_car(Train,Car2), car_color(Car2, blue).
eastbound(Train):- has_car(Train,Car), short(Car), has_car(Train,Car2), long(Car2), has_load(Car, A), has_load(Car2, A).