% Stephen Muggleton's Prolog code for
% randomly generating Michalski trains.
% adopted by Lukas Helff
% To run this you need a Prolog interpreter which executes
% a goal of the form:
% R is random
% Otherwise replace the definition of random/2 appropriately.
% (``R is random '' binds R to a pseudo-random number--floating
% point--between 0 and 1.)
% The top-level predicates are trains/0 and trains/1.
% modified for SWI-Prolog by Lukas Helff

loop(N_trains) :- N_trains>0, trains, C_train is N_trains-1, loop(C_train).

trains :- train1(X), show(X).

trains([]) :- !.
trains([H|T]) :- train1(H), randdirection(H), trains(T), !.
train1(Carriages) :-
  random([0,0.3,0.3,0.4],NCarriages),
  length(Carriages,NCarriages),
  carriages(Carriages,1),!.

randdirection(Carriages) :-
  random(0,2,Elem1),
  (Elem1 = 0;  eastbound(Carriages)), !.

carriages([],_).
carriages([C|Cs],N) :-
  carriage(C,N), N1 is N+1,
  carriages(Cs,N1), !.

carriage(c(N,Shape,Length,Double,Roof,Wheels,Load),N) :-
  randprop(car_length,[0.7,0.3],Length),
  shape(Length,Shape),
  double(Length,Shape,Double),
  roof1(Length,Shape,Roof),
  wheels(Length,Wheels),
  load(Length,Load), !.

shape(long,rectangle).
shape(short,S) :-
  randprop(car_shape,[0.048,0.048,0.524,0.190,0.190],S).

double(short,rectangle,Double) :-
  randprop(car_double,[0.73,0.27],Double), !.
double(_,_,not_double) :- !.

roof1(short,ellipse,arc) :- !.
roof1(short,hexagon,flat) :- !.
roof1(short,_,R) :- randprop(roof_shape,[0.842,0.105,0,0.053,0],R).
roof1(long,_,R) :- randprop(roof_shape,[0.333,0.444,0.223,0,0],R).

wheels(short,2).
wheels(long,W) :- random([0,0.56,0.44],W).

load(short,l(Shape,N)) :-
  randprop(load_shape,[0.381,0.048,0,0.190,0.381,0],Shape),
  random([0.952,0.048],N).

load(long,l(Shape,N)) :-
  randprop(load_shape,[0.125,0,0.125,0.625,0,0.125],Shape),
  random([0.11,0.55,0.11,0.23],N1), N is N1-1.

random(Dist,N) :-
  %R is random
  % R is 1,
  random(0,1.0,R),
  random(1,0,R,Dist,N).

random(N,_,_,[_],N).
random(N,P0,R,[P|_],N) :-
  P1 is P+P0, R=<P1, !.
random(N,P0,R,[P|Rest],M) :-
  P1 is P+P0, N1 is N+1,
  random(N1,P1,R,Rest,M), !.

randprop(Prop,Dist,Value) :-
  random(Dist,R),
  Call=..[Prop,R,Value],
  Call, !.

car_shape(1,ellipse). car_shape(2,hexagon).
car_shape(3,rectangle). car_shape(4,u_shaped). car_shape(5,bucket).
car_length(1,short). car_length(2,long).
car_open(1,open). car_open(2,closed).
car_double(1,not_double). car_double(2,double).
roof_shape(1,none). roof_shape(2,flat). roof_shape(3,jagged).
roof_shape(4,peaked). roof_shape(5,arc).
load_shape(1,circle). load_shape(2,diamond). load_shape(3,hexagon).
load_shape(4,rectangle). load_shape(5,triangle). load_shape(6,utriangle).

show(Train) :-
  direction(Train),
  show0(Train),
  open('raw/tmp/MichalskiTrains.txt', append, OS),
  format(OS, "~n" , []),
  close(OS),
  nl, !.

show0([]).
show0([C|Cs]) :-
  C=c(N,Shape,Length,Double,Roof,Wheels,l(Lshape,Lno)),
  open('raw/tmp/MichalskiTrains.txt', append, OS),
  format(OS, ' ~w ~w ~w ~w ~w ~w ~w ~w', [N,Shape,Length,Double,Roof,Wheels,Lshape,Lno]),
  writes(['Car ',N,': Shape = ',Shape,
  ', Length = ',Length,', Double = ',Double,nl, tab(8),
  'Roof = ',Roof,
  ', Wheels = ',Wheels,
  ', Load = ',Lno,' of ',Lshape,nl]),
  close(OS),
  show0(Cs), !.

direction(Train) :-
  open('raw/tmp/MichalskiTrains.txt', append, OS),
  (eastbound(Train) -> (format(OS, "~w" , [east]), write('Eastbound train:'), nl)
  ;otherwise -> (format(OS, "~w" , [west]), write('Westbound train:'), nl)),
  close(OS).

writes([]).
writes([H|T]) :-
  mywrite(H),
  writes(T).

mywrite(nl) :- nl, !.
mywrite(tab(X)) :- tab(X), !.
mywrite(X) :- write(X), !.

% Concept tester below emulates Michalski predicates.

% Theory X
% There is either a short, closed car, or a car with a circular load somewhere behind a car with a triangular load.
% eastbound([Car|Cars]):-
% (short(Car), closed(Car));
% (has_load0(Car,triangle), has_load1(Cars,circle));
% eastbound(Cars).


has_car(T,C) :- member(C,T).
has_car(T,C,N) :- member(C,T), car_num(C,N).

infront(T,C1,C2) :- append(_,[C1,C2|_],T).

ellipse(C) :- arg(2,C,ellipse).
hexagon(C) :- arg(2,C,hexagon).
rectangle(C) :- arg(2,C,rectangle).
u_shaped(C) :- arg(2,C,u_shaped).
bucket(C) :- arg(2,C,bucket).

long(C) :- arg(3,C,long).
short(C) :- arg(3,C,short).

double(C) :- arg(4,C,double).

has_roof(C,r(R,N)) :- arg(1,C,N), arg(5,C,R).
has_roof0(C,R) :- arg(5,C,R).

open(C) :- arg(5,C,none).
closed(C) :- not(open(C)).

has_wheel(C,w(NC,W)) :- arg(1,C,NC), arg(6,C,NW), nlist(1,NW,L), member(W,L).
has_wheel0(C,W) :- arg(6,C,W).

has_load(C,Load) :- arg(7,C,l(_,NLoad)), nlist(1,NLoad,L), member(Load,L).
has_load0(C,Shape) :- arg(7,C,l(Shape,N)), 1=<N.
has_load1(T,Shape) :- has_car(T,C), has_load0(C,Shape).

load_num(C, N) :- arg(7,C,l(_,N)).
car_num(C,N) :- arg(1,C,N).

none(r(none,_)). flat(r(flat,_)).
jagged(r(jagged,_)). peaked(r(peaked,_)).
arc(r(arc,_)).

member(X,[X|_]).
member(X,[_|T]) :- member(X,T).

nlist(N,N,[N]) :- !.
nlist(M,N,[M|T]) :-
  M=<N,
  M1 is M+1, nlist(M1,N,T), !.

len1([],0) :- !.
len1([_|T],N) :- len1(T,N1), N is N1+1, !.

append([],L,L) :- !.
append([H|L1],L2,[H|L3]) :-
  append(L1,L2,L3), !.

grey(C) :- car_color(C,grey).
red(C) :- car_color(C,red).
yellow(C) :- car_color(C,yellow).
blue(C) :- car_color(C,blue).
green(C) :- car_color(C,green).

car_color(C,CC) :- grey = CC, arg(2,C,ellipse).
car_color(C,CC) :- red = CC, arg(2,C,hexagon).
car_color(C,CC) :- yellow = CC, arg(2,C,rectangle).
car_color(C,CC) :- blue = CC, arg(2,C,u_shaped).
car_color(C,CC) :- green = CC, arg(2,C,bucket).

box(C) :- has_payload(C,box).
golden_vase(C) :- has_payload(C,golden_vase).
barrel(C) :- has_payload(C,barrel).
diamond(C) :- has_payload(C,diamond).
metal_pot(C) :- has_payload(C,metal_pot).
oval_vase(C) :- has_payload(C,oval_vase).

has_payload(C,Shape) :- box = Shape, arg(7,C,rectangle).
has_payload(C,Shape) :- golden_vase = Shape, arg(7,C,triangle).
has_payload(C,Shape) :- barrel = Shape, arg(7,C,circle).
has_payload(C,Shape) :- diamond = Shape, arg(7,C,diamond).
has_payload(C,Shape) :- metal_pot = Shape, arg(7,C,hexagon).
has_payload(C,Shape) :- oval_vase = Shape, arg(7,C,u_triangle).

roof_open(C) :- has_roof2(C,none).
roof_foundation(C) :- has_roof2(C,roof_foundation).
solid_roof(C) :- has_roof2(C,solid_roof).
braced_roof(C) :- has_roof2(C,braced_roof).
peaked_roof(C) :- has_roof2(C,peaked_roof).

has_roof2(C,Roof) :- none = Roof, arg(5,C,none).
has_roof2(C,Roof) :- roof_foundation = Roof, arg(5,C,arc).
has_roof2(C,Roof) :- solid_roof = Roof, arg(5,C,flat).
has_roof2(C,Roof) :- braced_roof = Roof, arg(5,C,jagged).
has_roof2(C,Roof) :- peaked_roof = Roof, arg(5,C,peaked).

has_roof2(C,open_roof(R)) :- arg(5,C,none).
has_roof2(C,roof_foundation(R)) :- arg(5,C,arc).
has_roof2(C,solid_roof(R)) :- arg(5,C,flat).
has_roof2(C,braced_roof(R)) :- arg(5,C,jagged).
has_roof2(C,peaked_roof(R)) :- arg(5,C,peaked).



solid_wall(C) :- arg(4,C,not_double).
braced_wall(C) :- arg(4,C,double).


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