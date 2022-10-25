

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


eastbound(A):- has_car(A,C),has_payload(C,D),oval_vase(D),car_num(C,B),has_wheel0(C,B).
eastbound(A):- has_car(A,C),roof_open(C),has_payload(C,B),barrel(B),long(C).
eastbound(A):- has_car(A,C),peaked_roof(C),solid_wall(C),has_payload(C,B),golden_vase(B).
eastbound(A):- has_car(A,B),has_car(A,C),braced_roof(C),has_payload(B,D),oval_vase(D).
eastbound(A):- has_car(A,C),has_wheel0(C,B),load_num(C,B),solid_wall(C),has_payload(C,D),barrel(D).
eastbound(A):- has_car(A,C),braced_wall(C),has_car(A,B),has_payload(B,D),blue(B),box(D).
eastbound(A):- has_car(A,B),has_payload(B,D),braced_wall(B),box(D),has_car(A,C),green(C).
eastbound(A):- has_car(A,C),short(C),load_num(C,D),car_num(C,D),has_car(A,B),roof_foundation(B).
eastbound(A):- has_car(A,B),long(B),load_num(B,D),has_car(A,C),car_num(C,D),braced_wall(C).
eastbound(A):- has_car(A,D),solid_wall(D),load_num(D,B),has_car(A,C),solid_roof(C),has_wheel0(C,B).
eastbound(A):- has_car(A,B),green(B),has_car(A,C),blue(C),has_car(A,D),yellow(D).
eastbound(A):- has_car(A,C),has_payload(C,D),has_wheel0(C,B),box(D),braced_roof(C),car_num(C,B).
eastbound(A):- has_car(A,C),long(C),has_car(A,B),has_payload(B,D),braced_wall(B),barrel(D).











