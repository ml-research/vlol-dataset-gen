import json
import math

import jsonpickle


class MichalskiTrain(object):
    def __init__(self, m_cars, direction, angle, scale=(0.5, 0.5, 0.5)):
        self.m_cars = m_cars
        self.direction = direction
        self.angle = int(angle)
        # blender scale of train
        self.scale = scale

        index = 1
        for car in m_cars:
            indicies = {}
            indicies["car"] = index
            index += 1
            indicies["wall"] = index
            index += 1
            indicies["roof"] = index
            index += 1
            indicies["wheels"] = index
            index += 1
            for load_number in range(car.get_load_number()):
                indicies["payload" + str(load_number)] = index
                index += 1
            car.set_index(indicies)
        # train rotation x axis = -.125
        self.blender_init_rotation = math.radians(-.125), 0, math.radians(90)

    def toJSON(self):
        return jsonpickle.encode(self)
        # return json.dumps(self, default=lambda o: o.__dict__,
        #                   sort_keys=True, indent=4)

    def get_cars(self):
        return self.m_cars

    def get_label(self):
        return self.direction

    def get_angle(self):
        return self.angle

    def get_blender_scale(self):
        return self.scale

    def get_car_length(self, car_part):
        # scalar length of individual car types
        car_length_scalar = {
            # short train length (2,9 - 0,909) = 1,991 along x axis
            "long": 3.54121375 * self.scale[0],
            # long train length (2,9 - 0.067029) = 2,832971 along x axis
            "short": 2.48875 * self.scale[0],
            # engine length (2,9 - 0.067029) = 2,832971 along x axis
            "engine": 3.75 * self.scale[0]
        }
        return car_length_scalar.get(car_part)

    def get_init_rotation(self):
        return self.blender_init_rotation


class MichalskiCar(object):
    def __init__(self, n, shape, length, double, roof, wheels, l_num, l_shape, scale=(0.5, 0.5, 0.5)):
        self.index = None
        self.blender_cords = {}
        self.n, self.shape, self.length, self.double, self.roof, self.wheels, self.l_num, self.l_shape, self.scale = int(
            n), shape, length, double, roof, int(wheels), int(l_num), l_shape, scale
        self.car_length_scalar = {
            # short train length (2,9 - 0,909) = 1,991 along x axis
            "long": 3.54121375 * self.scale[0],
            # long train length (2,9 - 0.067029) = 2,832971 along x axis
            "short": 2.48875 * self.scale[0],
            # engine length (2,9 - 0.067029) = 2,832971 along x axis
            "engine": 3.75 * self.scale[0]
        }[self.length]

        self.payload_scale = {
            "golden_vase": (0.015, 0.015, 0.05),
            "barrel": (0.4, 0.4, 0.4),
            "diamond": (40, 40, 40),
            "metal_pot": (1.2, 1.2, 1.2),
            "oval_vase": (2.5, 2.5, 2.5),
        }
        self.init_rotation = {
            "barrel": (0, 0, math.radians(90)),
            "oval_vase": (0, 0, math.radians(90)),
        }

    # get michalski attributes
    def get_car_number(self):
        return self.n

    def get_car_shape(self):
        return self.shape

    def get_car_length(self):
        return self.length

    def get_car_roof(self):
        return self.roof

    def get_car_wall(self):
        return self.double

    def get_car_wheels(self):
        return str(self.wheels) + '_wheels'

    def get_wheel_count(self):
        return self.wheels

    def get_load_number(self):
        return self.l_num

    def get_load_shape(self):
        return self.l_shape

    # get blender m_train attributes
    # michalski car shape is blender car color
    def get_blender_car_color(self):
        car_shape_to_material = {
            'rectangle': 'yellow',
            'bucket': 'green',
            'ellipse': 'grey',
            'hexagon': 'red',
            'u_shaped': 'blue',
        }
        return car_shape_to_material[self.shape]

    # the Michalski roof attribute can be represented as a blender train roof or modification on the car wall
    def get_blender_roof(self):
        roof_to_b_obj = {
            "none": 'none',
            "arc": "roof_foundation",
            "flat": 'solid_roof',
            "jagged": 'braced_roof',
            "peaked": 'peaked_roof'
        }
        return roof_to_b_obj[self.roof]

    def get_blender_wall(self):
        # roof_to_wall = {
        #     "none": None,
        #     "arc": "striped",
        #     "flat": 'full',
        #     "jagged": 'cage',
        #     "peaked": 'minimal'
        # }
        roof_to_wall = {
            "double": "braced_wall",
            "not_double": 'solid_wall',
        }
        return roof_to_wall[self.double]

    def get_blender_payload(self):
        load_to_b_obj = {
            "diamond": 'diamond',
            "rectangle": "box",
            "triangle": "golden_vase",
            "circle": 'barrel',
            "hexagon": 'metal_pot',
            "utriangle": 'oval_vase'
        }
        return load_to_b_obj[self.l_shape]

    def get_all_michalski_att(self):
        return self.shape, self.length, self.double, self.roof, self.wheels, self.l_num, self.l_shape

    def get_all_blender_att(self):
        return self.get_blender_car_color(), self.get_car_length(), self.get_blender_wall(), self.get_blender_roof(),\
               self.get_car_wheels(), self.get_load_number(), self.get_blender_payload()

    # attribute index used for blender segmentation
    def set_index(self, index):
        self.index = index

    def get_index(self, at):
        return self.index[at]

    # scale of car
    def get_scale(self):
        return self.scale

    def get_payload_scale(self):
        return self.payload_scale.get(self.get_blender_payload(), (1, 1, 1))

    def get_payload_rotation(self):
        return self.init_rotation.get(self.get_blender_payload(), (0, 0, 0))

    def get_car_length_scalar(self):
        return self.car_length_scalar

    # set and get blender world coords for individual car objects

    def set_blender_world_cord(self, obj_name, cord):
        self.blender_cords[obj_name] = cord

    def get_blender_world_cord(self, obj_name):
        return self.blender_cords[obj_name]
