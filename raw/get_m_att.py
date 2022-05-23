import read_raw_trains

def generate_m_train_attr_overview():
    double = []
    l_num = []
    l_shape = []
    length = []
    car_position = []
    roof = []
    shape = []
    wheels = []
    trains = util.read_trains()
    for train in trains:
        for car in train.m_cars:
            if car.double not in double:
                double.append(car.double)
            if car.l_num not in l_num:
                l_num.append(car.l_num)
            if car.l_shape not in l_shape:
                l_shape.append(car.l_shape)
            if car.length not in length:
                length.append(car.length)
            if car.n not in car_position:
                car_position.append(car.n)
            if car.roof not in roof:
                roof.append(car.roof)
            if car.shape not in shape:
                shape.append(car.shape)
            if car.wheels not in wheels:
                wheels.append(car.wheels)

    with open("class_att.py", "w+") as text_file:
        text_file.write("double values:\n %s\n" % double)
        text_file.write("load numbers:\n %s\n" % l_num)
        text_file.write("load shapes:\n %s\n" % l_shape)
        text_file.write("length:\n %s\n" % length)
        text_file.write("car positions:\n %s\n" % car_position)
        text_file.write("roofs:\n %s\n" % roof)
        text_file.write("shapes:\n %s\n" % shape)
        text_file.write("wheels:\n %s\n" % wheels)
