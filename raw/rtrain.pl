% Random generator of Michalski trains.

trains :-
	repeat, train1(X), show(X),
	write('More (y/n)? '),
	read(n).

trains([]) :- !.
trains([H|T]) :- train1(H), trains(T), !.

train1(Carriages) :-
	random([0,0.3,0.3,0.4],NCarriages),
	len1(Carriages,NCarriages),
	carriages(Carriages,1), !.

carriages([],_).
carriages([C|Cs],N) :-
	carriage(C,N), N1 is N+1,
	carriages(Cs,N1), !.

carriage(c(N,Shape,Length,Double,Roof,Wheels,Load),N) :-
	randprop(car_length,[0.7,0.3],Length),
	% randprop(car_double,[0.9,0.1],Double),
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

load(Length,Load) :-
	load1(Length,l(Shape,N)),
	((N=0 -> (Load=nil))
	 ;otherwise -> (Load=l(Shape,N))).

load1(short,l(Shape,N)) :-
	randprop(load_shape,[0.381,0.048,0,0.190,0.381,0],Shape),
	random([0.952,0.048],N).
load1(long,l(Shape,N)) :-
	randprop(load_shape,[0.125,0,0.125,0.625,0,0.125],Shape),
	random([0.11,0.55,0.11,0.23],N1), N is N1-1.

random(Dist,N) :-
	R is random,
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

car_shape(1,ellipse).  car_shape(2,hexagon).
car_shape(3,rectangle).  car_shape(4,u_shaped). car_shape(5,bucket).
car_length(1,short).  car_length(2,long).
car_open(1,open).  car_open(2,closed).
car_double(1,not_double).  car_double(2,double).
roof_shape(1,none).  roof_shape(2,flat).  roof_shape(3,jagged).
roof_shape(4,peaked). roof_shape(5,arc).
load_shape(1,circle).  load_shape(2,diamond).  load_shape(3,hexagon).
load_shape(4,rectangle).  load_shape(5,triangle).  load_shape(6,utriangle).

show(Train) :-
	nl,
	(eastbound(Train) -> (write('Eastbound train:'), nl)
	;otherwise -> (write('Westbound train:'),nl)),
	show0(Train), nl, !.

show0([]).
show0([C|Cs]) :-
	C=c(N,Shape,Length,Double,Roof,Wheels,l(Lshape,Lno)),
	writes(['Car ',N,': Shape = ',Shape,
		', Length = ',Length,', Double = ',Double,nl, tab(8),
		'Roof = ',Roof,
		', Wheels = ',Wheels,
		', Load = ',Lno,' of ',Lshape,nl]),
	show0(Cs), !.
show0([C|Cs]) :-
	C=c(N,Shape,Length,Double,Roof,Wheels,nil),
	writes(['Car ',N,': Shape = ',Shape,
		', Length = ',Length,', Double = ',Double,nl, tab(8),
		'Roof = ',Roof,
		', Wheels = ',Wheels,
		', Load = nil']),
	show0(Cs), !.

% writes([]).
% writes([H|T]) :-
%	mywrite(H),
%	writes(T).

mywrite(nl) :- nl, !.
mywrite(tab(X)) :- tab(X), !.
mywrite(X) :- write(X), !.

:- consult(background), consult(concept)?
