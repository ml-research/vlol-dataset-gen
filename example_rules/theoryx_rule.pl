% Default classification rule "Theory X"
% There is either a short, closed car, or a car with a circular load somewhere behind a car with a triangular load.
eastbound([Car|Cars]):- (short(Car), closed(Car)); (has_load0(Car,triangle), has_load1(Cars,circle)); eastbound(Cars).

