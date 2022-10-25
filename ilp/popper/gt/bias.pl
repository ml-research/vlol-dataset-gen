%% include some ground atoms but not all
max_vars(6).
max_body(6).

%% general
head_pred(eastbound,1).
body_pred(has_car,2).
%% car number
body_pred(car_num,2).
%% payload number
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
type(has_payload,(car,load)).
%% car number
type(car_num,(car,integer)).
%% colors
type(yellow,(car,)).
type(green,(car,)).
type(grey,(car,)).
type(red,(car,)).
type(blue,(car,)).
%% lengths
type(short,(car,)).
type(long,(car,)).
%% wheels
type(has_wheel0,(car,integer)).
%% roofs
type(roof_open,(car,)).
type(roof_foundation,(car,)).
type(solid_roof,(car,)).
type(braced_roof,(car,)).
type(peaked_roof,(car,)).
%% walls
type(braced_wall,(car,)).
type(solid_wall,(car,)).
%% payload number
type(load_num,(car,integer)).
%% payload shape
type(barrel,(load,)).
type(golden_vase,(load,)).
type(box,(load,)).
type(diamond,(load,)).
type(metal_pot,(load,)).
type(oval_vase,(load,)).

%% general
direction(eastbound,(in,)).
direction(has_car,(in,out)).
direction(has_payload,(in,out)).
%%direction(behind,(in,in)).
%% car number
direction(car_num,(in,out)).
%% colors
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
direction(roof_open,(in,)).
direction(roof_foundation,(in,)).
direction(solid_roof,(in,)).
direction(braced_roof,(in,)).
direction(peaked_roof,(in,)).
%% payload number
direction(load_num,(in,out)).
%% payload shape
direction(barrel,(in,)).
direction(golden_vase,(in,)).
direction(box,(in,)).
direction(diamond,(in,)).
direction(metal_pot,(in,)).
direction(oval_vase,(in,)).
%% walls
direction(braced_wall,(in,)).
direction(solid_wall,(in,)).
