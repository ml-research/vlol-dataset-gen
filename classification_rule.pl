% The classification rule can be adjusted according the requirements. Feel free to make any adjustments needed.
% The classification rule must be specified in the Prolog description language using the defined predicates (see README)
% new predicates can also be defined using the existing ones

% Default classification rule "Theory X"
% There is either a short, closed car, or a car with a circular load somewhere behind a car with a triangular load.

eastbound(Train):-
    (has_car(Train,Car), closed(Car), double(Car), load_num(Car,N), N>=1);
    (has_car(Train,Car), closed(Car), bucket(Car));
    (has_car(Train,Car), load_num(Car,N), has_wheel(Car,w(CN,N))), CN<N.

%eastbound([Car|Cars]):-
%(short(Car), closed(Car));
%(has_load0(Car,triangle), has_load1(Cars,circle));
%eastbound(Cars).

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