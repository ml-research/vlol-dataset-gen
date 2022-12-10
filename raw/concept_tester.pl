

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


%eastbound([Car|Cars]):- (short(Car), closed(Car)); (has_load0(Car,triangle), has_load1(Cars,circle)); eastbound(Cars).

eastbound(A) :-has_car(A,B), has_car(A,C), has_roof2(C,peaked_roof).
eastbound(A) :-has_car(A,B), has_car(A,C), short(C), has_roof2(C,solid_roof).
eastbound(A) :-has_car(A,B), car_color(B,grey).
eastbound(A) :-has_car(A,B), car_num(B,1), has_payload(B,golden_vase), has_car(A,C),has_payload(C,barrel).
eastbound(A) :-has_car(A,B), has_payload(B,golden_vase), has_car(A,C), car_num(C,4),has_payload(C,barrel).
eastbound(A) :-has_car(A,B), car_num(B,2), has_payload(B,golden_vase), has_car(A,C),car_num(C,3), has_payload(C,barrel).











