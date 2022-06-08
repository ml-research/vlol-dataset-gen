% the classification rule can be adjusted according the requirements
% classification rule must be specified in the Prolog description language

% Default classification rule "Theory X"
% There is either a short, closed car, or a car with a circular load somewhere behind a car with a triangular load.
eastbound([Car|Cars]):-
(short(Car), closed(Car));
(has_load0(Car,triangle), has_load1(Cars,circle));
eastbound(Cars).

% by default the descriptors listed below are available:
%has_car(T,C)
%infront(T,C1,C2)
%car_length(1,short)
%roof_shape(1,none)
%roof_shape(4,peaked)
%load_shape(1,circle)

