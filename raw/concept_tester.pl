

%eastbound([Car|Cars]):-
%  (short(Car), closed(Car));
%  (has_load0(Car,triangle), has_load1(Cars,circle));
%  eastbound(Cars).

eastbound(Train):-
    (has_car(Train,Car), closed(Car), double(Car), load_num(Car,N), N>=1);
    (has_car(Train,Car), closed(Car), bucket(Car));
    (has_car(Train,Car), load_num(Car,N), has_wheel(Car,w(CN,N))), CN<N.
