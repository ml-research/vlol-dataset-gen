% The classification rule can be adjusted according the requirements. Feel free to make any adjustments needed.
% The classification rule must be specified in the Prolog description language using the defined predicates (see README)
% new predicates can also be defined using the existing ones




% numerical rule
%eastbound(Train):- has_car(Train,Car), load_num(Car,N), car_num(Car,N), has_wheel0(Car,N).
%eastbound(Train):- has_car(Train,Car), load_num(Car, N), has_car(Train,Car2), has_wheel0(Car2, N), car_color(Car, A), car_color(Car2, A).

% easy rule
% train with 3 different car colors --> popper fails, popper is not able to perform numerical comparison e.g. x,y∈R x<y
%eastbound(A):- has_car(A,B), has_car(A,C), has_car(A,D), car_color(D,X), car_color(C,Y), car_color(B,Z), X\=Y, Y\=Z, Z\=X.

% train with a short and a long car which have the same color
%eastbound(Train):- has_car(Train,Car), has_car(Train,Car2), short(Car), long(Car2), car_color(Car, A), car_color(Car2, A).

% complex rule
%eastbound(Train):- has_car(Train,Car), load_num(Car,N), car_num(Car,N), has_wheel0(Car,N).
%eastbound(Train):- (has_car(Train,Car), closed(Car), (double(Car); bucket(Car))).
%eastbound(Train):- has_car(Train,Car), has_car(Train,Car2), has_car(Train,Car3), car_color(Car, A), car_color(Car3, A), car_color(Car2, A).


% train with braced_wall and 2 loads, load_num 3 and a blue car or a blue car and braced_wall --> popper fails
eastbound(Train):- has_car(Train,Car), braced_wall(Car), has_car(Train,Car2), load_num(Car2,2).
eastbound(Train):- has_car(Train,Car), load_num(Car,3), has_car(Train,Car2), car_color(Car2, blue).
eastbound(Train):- has_car(Train,Car), braced_wall(Car), has_car(Train,Car2), car_color(Car2, blue).


%eastbound(Train):- has_car(Train,Car2), car_color(Car2, grey).


% Default classification rule "Theory X"
% There is either a short, closed car, or a car with a circular load somewhere behind a car with a triangular load.
%eastbound([Car|Cars]):-
%    (short(Car), closed(Car));
%    (has_load0(Car,triangle), has_load1(Cars,circle));
%    eastbound(Cars).

%%%%%%%%%%%%%%%%%%%%%%%%
% Other example classification rules:

% The train has either a closed last car or a triangular load in a car other than the last one. Also, either the last
% car is short or the train, after the last car is removed, has the above property.
%eastbound([Car|Cars]) :-
%(closed(Car);has_load1(Cars, triangle)),
%(short(Car);eastbound(Cars)).

%The third car [from the end] has a triangular load, the second [to last] car is hexagon-shaped, or the last car is
%rectangularly shaped and the third [from the end] car is closed.
%eastbound([Car1,Car2,Car3|_]) :-
%has_load0(Car3,triangle);
%rectangle(Car1),(closed(Car3);
%hexagon(Car2)).

% example rules taken from https://www.researchgate.net/publication/2750403_How_Did_AQ_Face_The_East-West_Challenge_-_An_Analysis_of_the_AQ_Family%27s_Performance_in_the_2nd_International_Competition_of_Machine_Learning_Programs