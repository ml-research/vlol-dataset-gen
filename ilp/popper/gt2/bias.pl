%% Maximum expressiveness of the predicates
max_clauses(2).
max_vars(6).
max_body(6).

%% general
head_pred(eastbound,1).
body_pred(has_car,2).
body_pred(car_num,2).
%% payload
body_pred(has_payload,2).
body_pred(load_num,2).
%% payload shape
body_pred(barrel,1).
body_pred(golden_vase,1).
body_pred(box,1).
body_pred(diamond,1).
body_pred(metal_pot,1).
body_pred(oval_vase,1).
%% colors
body_pred(car_color,2).
body_pred(yellow,1).
body_pred(green,1).
body_pred(grey,1).
body_pred(red,1).
body_pred(blue,1).
%% lengths
body_pred(short,1).
body_pred(long,1).
%% wheels
body_pred(has_wheel0,2).
%% roofs
body_pred(has_roof2,2).
body_pred(roof_open,1).
body_pred(roof_foundation,1).
body_pred(solid_roof,1).
body_pred(braced_roof,1).
body_pred(peaked_roof,1).
%% walls
body_pred(braced_wall,1).
body_pred(solid_wall,1).

%% general
type(eastbound,(train,)).
type(has_car,(train,car)).
%% car number
type(car_num,(car,integer)).
%% payload
type(has_payload,(car,load)).
type(load_num,(car,integer)).
%% payload shape
type(barrel,(load,)).
type(golden_vase,(load,)).
type(box,(load,)).
type(diamond,(load,)).
type(metal_pot,(load,)).
type(oval_vase,(load,)).
%% colors
type(car_color,(car,color)).
type(yellow,(color,)).
type(green,(color,)).
type(grey,(color,)).
type(red,(color,)).
type(blue,(color,)).
%% lengths
type(short,(car,)).
type(long,(car,)).
%% wheels
type(has_wheel0,(car,integer)).
%% roofs
type(has_roof2,(car,roof)).
%% roofs
type(roof_open,(roof,)).
type(roof_foundation,(roof,)).
type(solid_roof,(roof,)).
type(braced_roof,(roof,)).
type(peaked_roof,(roof,)).
%% walls
type(braced_wall,(car,)).
type(solid_wall,(car,)).

%% general
direction(eastbound,(in,)).
direction(has_car,(in,out)).
%% car number
direction(car_num,(in,out)).
%% payload
direction(has_payload,(in,out)).
direction(load_num,(in,out)).
%% payload shape
direction(barrel,(in,)).
direction(golden_vase,(in,)).
direction(box,(in,)).
direction(diamond,(in,)).
direction(metal_pot,(in,)).
direction(oval_vase,(in,)).
%% colors
direction(car_color,(in,out)).
direction(yellow,(in,)).
direction(green,(in,)).
direction(grey,(in,)).
direction(red,(in,)).
direction(blue,(in,)).
%% lengths
direction(short,(in,)).
direction(long,(in,)).
%% wheels
direction(has_wheel0,(in,out)).
%% roofs
direction(has_roof2,(in,out)).
direction(roof_open,(in,)).
direction(roof_foundation,(in,)).
direction(solid_roof,(in,)).
direction(braced_roof,(in,)).
direction(peaked_roof,(in,)).
%% walls
direction(braced_wall,(in,)).
direction(solid_wall,(in,)).
