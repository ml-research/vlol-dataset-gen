

%eastbound([Car|Cars]):-
%  (short(Car), closed(Car));
%  (has_load0(Car,triangle), has_load1(Cars,circle));
%  eastbound(Cars).

eastbound(Train):-
    (has_car(Train,Car), closed(Car), (double(Car); bucket(Car)));
    (has_car(Train,Car), load_num(Car,3), has_wheel(Car,w(NC,3))).
