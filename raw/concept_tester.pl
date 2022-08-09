direction(Train) :-
  open('raw/concept_tester.txt', append, OS),
  (eastbound(Train) -> (format(OS, "~w" , [east]), write('Eastbound train'), nl)
  ;otherwise -> (format(OS, "~w" , [west]), write('Westbound train'), nl)),
  close(OS).

%eastbound([Car|Cars]):-
%  (short(Car), closed(Car));
%  (has_load0(Car,triangle), has_load1(Cars,circle));
%  eastbound(Cars).

eastbound([Car|Cars]):-
  (short(Car), closed(Car));
  ellipse(Car);
  eastbound(Cars).


has_car(T,C) :- member(C,T).

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

open(C) :- arg(5,C,none).
closed(C) :- not(open(C)).

has_wheel(C,w(NC,W)) :- arg(1,C,NC), arg(6,C,NW), nlist(1,NW,L), member(W,L).

has_load(C,Load) :- arg(7,C,l(_,NLoad)), nlist(1,NLoad,L), member(Load,L).
has_load0(C,Shape) :- arg(7,C,l(Shape,N)), 1=<N.
has_load1(T,Shape) :- has_car(T,C), has_load0(C,Shape).

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
