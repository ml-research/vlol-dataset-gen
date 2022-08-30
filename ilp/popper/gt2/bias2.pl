
max_clauses(2).
max_vars(6).
max_body(6).

%% general
head_pred(f,1).
body_pred(has_car,2).
body_pred(car_num,2).
%% payload
body_pred(has_payload,2).
body_pred(load_num,2).
%% colors
body_pred(car_color,2).
%% lengths
body_pred(short,1).
body_pred(long,1).
%% wheels
body_pred(has_wheel0,2).
%% roofs
body_pred(has_roof0,2).
%% walls
body_pred(braced_wall,1).
body_pred(solid_wall,1).

%% general
type(f,(train,)).
type(has_car,(train,car)).
type(car_num,(car,int)).
%% payload
type(has_payload,(car,shape)).
type(load_num,(car,integer)).
%% colors
type(car_color,(car,color)).
%% lengths
type(short,(car,)).
type(long,(car,)).
%% wheels
type(has_wheel0,(car,integer)).
%% roofs
type(has_roof0,(car,roof)).
%% walls
type(braced_wall,(car,)).
type(solid_wall,(car,)).

%% general
direction(f,(in,)).
direction(has_car,(in,out)).
direction(car_num,(in,out)).
%% car number
direction(has_payload,(in,out)).
direction(load_num,(in,out)).
%% colors
direction(car_color,(in,out)).
%% lengths
direction(short,(in,)).
direction(long,(in,)).
%% wheels
direction(has_wheel0,(in,out)).
%% roofs
direction(has_roof0,(in,out)).
%% walls
direction(braced_wall,(in,)).
direction(solid_wall,(in,)).
