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

loop(N_trains,NCarriages) :- N_trains>0, trains(NCarriages), C_train is N_trains-1, loop(C_train).

trains(NCarriages) :- train1(X, NCarriages), show(X).

%trains([]) :- !.
%trains([H|T]) :- train1(H, NCarriages), randdirection(H), trains(T), !.
train1(Carriages, NCarriages) :-
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
yellow(C) :- car_color(C,yellow).
red(C) :- car_color(C,red).
blue(C) :- car_color(C,blue).
green(C) :- car_color(C,green).

roof_open(C) :- has_roof2(C,none).
roof_foundation(C) :- has_roof2(C,roof_foundation).
solid_roof(C) :- has_roof2(C,solid_roof).
braced_roof(C) :- has_roof2(C,braced_roof).
peaked_roof(C) :- has_roof2(C,peaked_roof).

box(C) :- has_payload(C,box).
golden_vase(C) :- has_payload(C,golden_vase).
barrel(C) :- has_payload(C,barrel).
diamond(C) :- has_payload(C,diamond).
metal_pot(C) :- has_payload(C,metal_pot).
oval_vase(C) :- has_payload(C,oval_vase).

car_color(C,grey) :- arg(2,C,ellipse).
car_color(C,red) :- arg(2,C,hexagon).
car_color(C,yellow) :- arg(2,C,rectangle).
car_color(C,blue) :- arg(2,C,u_shaped).
car_color(C,green) :- arg(2,C,bucket).

has_payload(C,box) :- arg(7,C,l(rectangle,N)), 1=<N.
has_payload(C,golden_vase) :- arg(7,C,l(triangle,N)), 1=<N.
has_payload(C,barrel) :- arg(7,C,l(circle,N)), 1=<N.
has_payload(C,diamond) :- arg(7,C,l(diamond,N)), 1=<N.
has_payload(C,metal_pot) :- arg(7,C,l(hexagon,N)), 1=<N.
has_payload(C,oval_vase) :- arg(7,C,l(utriangle,N)), 1=<N.

has_roof2(C,none) :- arg(5,C,none).
has_roof2(C,roof_foundation) :- arg(5,C,arc).
has_roof2(C,solid_roof) :- arg(5,C,flat).
has_roof2(C,braced_roof) :- arg(5,C,jagged).
has_roof2(C,peaked_roof) :- arg(5,C,peaked).

roof_open3(none).
roof_foundation3(roof_foundation).
solid_roof3(solid_roof).
braced_roof3(braced_roof).
peaked_roof3(peaked_roof).

grey3(grey).
yellow3(yellow).
red3(red).
blue3(blue).
green3(green).

box3(box).
golden_vase3(golden_vase).
barrel3(barrel).
diamond3(diamond).
metal_pot3(metal_pot).
oval_vase3(oval_vase).

car_color3(C,grey(_R)) :- arg(2,C,ellipse).
car_color3(C,red(_R)) :- arg(2,C,hexagon).
car_color3(C,yellow(_R)) :- arg(2,C,rectangle).
car_color3(C,blue(_R)) :- arg(2,C,u_shaped).
car_color3(C,green(_R)) :- arg(2,C,bucket).

has_payload3(C,box(_R)) :- arg(7,C,rectangle).
has_payload3(C,golden_vase(_R)) :- arg(7,C,triangle).
has_payload3(C,barrel(_R)) :- arg(7,C,circle).
has_payload3(C,diamond(_R)) :- arg(7,C,diamond).
has_payload3(C,metal_pot(_R)) :- arg(7,C,hexagon).
has_payload3(C,oval_vase(_R)) :- arg(7,C,utriangle).

has_roof3(C,open_roof(_R)) :- arg(5,C,none).
has_roof3(C,roof_foundation(_R)) :- arg(5,C,arc).
has_roof3(C,solid_roof(_R)) :- arg(5,C,flat).
has_roof3(C,braced_roof(_R)) :- arg(5,C,jagged).
has_roof3(C,peaked_roof(_R)) :- arg(5,C,peaked).

solid_wall(C) :- arg(4,C,not_double).
braced_wall(C) :- arg(4,C,double).
