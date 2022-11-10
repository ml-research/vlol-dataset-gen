% color rule:
% A Train with 3 different car colors

eastbound(A):- has_car(A,B), has_car(A,C), has_car(A,D), car_color(D,X), car_color(C,Y), car_color(B,Z), X\=Y, Y\=Z, Z\=X.