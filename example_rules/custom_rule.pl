% A costum rule:
% The costum classification rule can be adjusted according the requirements. Feel free to make any adjustments needed.
% The classification rule must be specified in the Prolog description language using the defined predicates (see README).
% New predicates can also be used if they are defined in below (structure of the trains can be found in train_generator.pl file)

eastbound(Train):- has_car(Train,Car), load_num(Car,N), car_num(Car,N), has_wheel0(Car,N).
