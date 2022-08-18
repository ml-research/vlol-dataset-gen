max_clauses(2).
max_vars(6).
max_body(6).

%% general
head_pred(f,1).
body_pred(has_car,2).
body_pred(has_load,2).
%%body_pred(behind,2).
%% car number
body_pred(first_car,1).
body_pred(second_car,1).
body_pred(third_car,1).
body_pred(fourth_car,1).
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
body_pred(two_wheels,1).
body_pred(three_wheels,1).
%% roofs
body_pred(roof_open,1).
body_pred(roof_closed,1).
body_pred(roof_foundation,1).
body_pred(solid_roof,1).
body_pred(braced_roof,1).
body_pred(peaked_roof,1).
%% walls
body_pred(braced_wall,1).
body_pred(solid_wall,1).
%% payload number
body_pred(zero_load,1).
body_pred(one_load,1).
body_pred(two_load,1).
body_pred(three_load,1).
%% payload shape
body_pred(barrel,1).
body_pred(golden_vase,1).
body_pred(box,1).
body_pred(diamond,1).
body_pred(metal_pot,1).
body_pred(oval_vase,1).


%% general
type(f,(train,)).
type(has_car,(train,car)).
type(has_load,(car,load)).
%type(behind,(car,car)).
%% car number
type(first_car,(car,)).
type(second_car,(car,)).
type(third_car,(car,)).
type(fourth_car,(car,)).
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
type(two_wheels,(car,)).
type(three_wheels,(car,)).
%% roofs
type(roof_open,(car,)).
type(roof_closed,(car,)).
type(roof_foundation,(car,)).
type(solid_roof,(car,)).
type(braced_roof,(car,)).
type(peaked_roof,(car,)).
%% walls
type(braced_wall,(car,)).
type(solid_wall,(car,)).
%% payload number
type(zero_load,(car,)).
type(one_load,(car,)).
type(two_load,(car,)).
type(three_load,(car,)).
%% payload shape
type(barrel,(load,)).
type(golden_vase,(load,)).
type(box,(load,)).
type(diamond,(load,)).
type(metal_pot,(load,)).
type(oval_vase,(load,)).

%% general
direction(f,(in,)).
direction(has_car,(in,out)).
direction(has_load,(in,out)).
%%direction(behind,(in,out)).
%% car number
direction(first_car,(in,)).
direction(second_car,(in,)).
direction(third_car,(in,)).
direction(fourth_car,(in,)).
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
direction(two_wheels,(in,)).
direction(three_wheels,(in,)).
%% roofs
direction(roof_open,(in,)).
direction(roof_closed,(in,)).
direction(roof_foundation,(in,)).
direction(solid_roof,(in,)).
direction(braced_roof,(in,)).
direction(peaked_roof,(in,)).
%% payload number
direction(zero_load,(in,)).
direction(one_load,(in,)).
direction(two_load,(in,)).
direction(three_load,(in,)).
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
