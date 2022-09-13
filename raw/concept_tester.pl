

%eastbound([Car|Cars]):-
%  (short(Car), closed(Car));
%  (has_load0(Car,triangle), has_load1(Cars,circle));
%  eastbound(Cars).

%eastbound(Train):-
%    (has_car(Train,Car), closed(Car), (double(Car); bucket(Car)));
%    (has_car(Train,Car), load_num(Car,3), has_wheel(Car,w(NC,3))).

%eastbound(A) :-
%   has_car(A,B,1), has_car(A,C,2), has_car(A,D,3), car_color(D,yellow),
%   car_color(C,yellow), car_color(B,yellow), has_wheel0(D,2), has_wheel0(C,3),
%   has_wheel0(B,3), has_roof0(D,none), has_roof0(C,solid_roof), has_roof0(B,none).

%eastbound(Train):- has_car(Train,Car), braced_wall(Car), has_car(Train,Car2), load_num(Car2, 2).
%eastbound(Train):- has_car(Train,Car), load_num(Car,3), has_car(Train,Car2), car_color(Car2, blue).
%eastbound(Train):- has_car(Train,Car), short(Car), has_car(Train,Car2), long(Car2), has_load(Car, A), has_load(Car2, A).

eastbound(A):- has_car(A,C),has_payload(C,B),load_num(C,D),oval_vase(B),has_wheel0(C,D).
eastbound(A):- has_car(A,C),has_car(A,D),braced_wall(D),has_wheel0(D,B),load_num(C,B).
eastbound(A):- has_car(A,D),has_car(A,C),braced_wall(C),car_color(D,B),blue(B).
eastbound(A):- has_car(A,E),has_payload(E,B),diamond(B),has_car(A,C),has_wheel0(C,D),load_num(C,D).
eastbound(A):- has_car(A,C),has_wheel0(C,D),has_payload(C,B),car_num(C,D),long(C),barrel(B).