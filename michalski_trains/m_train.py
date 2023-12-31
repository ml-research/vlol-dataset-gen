import math
from dataclasses import dataclass

import jsonpickle


@dataclass
class MichalskiTrain(object):
    '''
    A Michalski train object that contains the train data (either a list of BlenderCar objects or a list of SimpleCar objects)
    i.e. a list of MichalskiCar objects and the train label and angle of rotation as well as its scale.
    '''

    def __init__(self, m_cars, direction, angle, scale=(0.5, 0.5, 0.5)):
        '''
        @param m_cars: list of MichalskiCar objects or string containing the train description
        @param direction: direction of train
        @param angle: angle of train rotation
        @param scale: scale of train (x,y,z) axis
        '''
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

    @classmethod
    def from_text(cls, text, visualization):
        cars = []
        l = text.split(' ')
        dir = l[0]
        t_angle = l[1]
        train_length = len(l) // 8
        for c in range(train_length):
            ind = c * 8
            car = BlenderCar(l[ind + 2], l[ind + 3], l[ind + 4], l[ind + 5], l[ind + 6], l[ind + 7], l[ind + 8],
                             l[ind + 9].strip('\n'))
            if visualization == 'block':
                car = SimpleCar(l[ind + 2], l[ind + 3], l[ind + 4], l[ind + 5], l[ind + 6], l[ind + 7], l[ind + 8],
                                l[ind + 9].strip('\n'))

            cars.append(car)
        train = cls(cars, dir, t_angle)
        if visualization == 'block':
            train.update_pass_indices()
        return train

    @classmethod
    def fromJSON(self, obj):
        return jsonpickle.decode(obj)

    def toJSON(self):
        return jsonpickle.encode(self)
        # return json.dumps(self, default=lambda o: o.__dict__,
        #                   sort_keys=True, indent=4)

    def get_cars(self):
        return self.m_cars

    def get_label(self):
        return self.direction

    def set_label(self, label):
        self.direction = label

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
            "engine": 3.75 * self.scale[0],
            # short train length
            "simple_long": 3 * self.scale[0],
            # long train length
            "simple_short": 2 * self.scale[0],
            # engine length
            "simple_engine": 3.2 * self.scale[0]
        }
        return car_length_scalar.get(car_part)

    def get_init_rotation(self):
        return self.blender_init_rotation

    def update_pass_indices(self):
        index = 1
        for car in self.m_cars:
            indicies = {"car": index, "roof": index}
            index += 1
            indicies["wall"] = index
            index += 1
            indicies["wheels"] = index
            index += 1
            for load_number in range(car.get_load_number()):
                indicies["payload" + str(load_number)] = index
                index += 1
            car.set_index(indicies)

    def to_txt(self):
        txt = f'{self.direction} {self.angle}'
        for car in self.m_cars:
            txt += ' ' + car.to_txt()
        return txt


@dataclass
class MichalskiCar(object):
    '''
    original Michalski car object that contains the original attributes and values
    car number, shape, length, wall, roof, wall, wheels, payload number, and payload shape
    '''

    def __init__(self, n, shape, length, double, roof, wheels, l_shape, l_num):
        '''
        @param n: car number
        @param shape: car shape
        @param length: car length
        @param double: car wall
        @param roof: car roof
        @param wheels: car wheels
        @param l_shape: payload shape
        @param l_num: payload number
        '''
        self.index = None
        self.blender_cords = {}
        self.n, self.shape, self.length, self.double, self.roof, self.wheels, self.l_num, self.l_shape = int(
            n), shape, length, double, roof, int(wheels), int(l_num), l_shape

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

    def to_txt(self):
        return str(self.n) + " " + self.shape + " " + self.length + " " + self.double + " " + self.roof + " " + str(
            self.wheels) + " " + self.l_shape + " " + str(self.l_num)


@dataclass
class BlenderCar(MichalskiCar):
    '''
    Blender car object that inherits from MichalskiCar and contains the blender attributes and values
    @param n: car number
    @param shape: car shape
    @param length: car length
    @param double: car wall
    @param roof: car roof
    @param wheels: car wheels
    @param l_shape: payload shape
    @param l_num: payload number
    '''

    def __init__(self, n, shape, length, double, roof, wheels, l_shape, l_num, scale=None):
        super().__init__(n, shape, length, double, roof, wheels, l_shape, l_num)
        self.scale = [.5, .5, .5] if scale is None else scale
        self.car_length_scalar = {
            # short train length (2,9 - 0,909) = 1,991 along x axis
            "long": 3.54121375 * self.scale[0],
            # long train length (2,9 - 0.067029) = 2,832971 along x axis
            "short": 2.48875 * self.scale[0],
            # engine length (2,9 - 0.067029) = 2,832971 along x axis
            "engine": 3.75 * self.scale[0]
        }[self.length]

        self.payload_scale = {
            "golden_vase": (0.03, 0.03, 0.1),
            "barrel": (0.8, 0.8, 0.8),
            "diamond": (80, 80, 80),
            "metal_pot": (2.4, 2.4, 2.4),
            "oval_vase": (5, 5, 5),
        }
        self.init_rotation = {
            "barrel": (0, 0, math.radians(90)),
            "oval_vase": (0, 0, math.radians(90)),
        }

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
            "utriangle": 'oval_vase',
            "none": 'none'
        }
        return load_to_b_obj[self.l_shape]

    def get_all_michalski_att(self):
        return self.shape, self.length, self.double, self.roof, self.wheels, self.l_num, self.l_shape

    def get_all_blender_att(self):
        return self.get_blender_car_color(), self.get_car_length(), self.get_blender_wall(), self.get_blender_roof(), \
            self.get_car_wheels(), self.get_load_number(), self.get_blender_payload()

    # attribute index used for blender segmentation
    def set_index(self, index):
        self.index = index

    def get_index(self, at):
        return self.index[at]

    # scale of car
    def get_blender_scale(self):
        return self.scale

    def get_payload_scale(self):
        scale = self.get_blender_scale()
        payload_init_scale = self.payload_scale.get(self.get_blender_payload(), (2, 2, 2))
        return scale[0] * payload_init_scale[0], scale[1] * payload_init_scale[1], scale[2] * payload_init_scale[2]
        # return self.payload_scale.get(self.get_blender_payload(), (1, 1, 1))

    def get_payload_rotation(self):
        return self.init_rotation.get(self.get_blender_payload(), (0, 0, 0))

    def get_car_length_scalar(self):
        return self.car_length_scalar

    # set and get blender world coords for individual car objects
    def set_blender_world_cord(self, obj_name, cord):
        self.blender_cords[obj_name] = cord

    def get_blender_world_cord(self, obj_name):
        return self.blender_cords[obj_name]


@dataclass
class SimpleCar(BlenderCar):
    '''
    Simple car object that inherits from BlenderCar and contains the simple attributes and values
    @param n: car number
    @param shape: car shape
    @param length: car length
    @param double: car wall
    @param roof: car roof
    @param wheels: car wheels
    @param l_shape: payload shape
    @param l_num: payload number
    '''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.car_length_scalar = {
            # long train length 3
            "long": 3 * self.scale[0],
            # short train length 2
            "short": 2 * self.scale[0],
            # engine length 3.2
            "engine": 3.2 * self.scale[0]
        }[self.length]

    def get_simple_color(self):
        car_shape_to_material = {
            'rectangle': 'yellow',
            'bucket': 'green',
            'ellipse': 'grey',
            'hexagon': 'red',
            'u_shaped': 'blue',
        }
        return car_shape_to_material[self.shape]

    def get_simple_platform_length(self):
        return self.get_car_length()

    def get_simple_side_object_frustum(self):
        roof_to_frustum = {
            "double": "down",
            "not_double": 'up',
        }
        return roof_to_frustum[self.double]

    def get_simple_platform_shape(self):
        roof_to_b_obj = {
            "none": 'cube',
            "arc": 'cylinder',
            "flat": 'hemisphere',
            "jagged": 'frustum',
            "peaked": 'hexagonal_prism'
        }
        return roof_to_b_obj[self.roof]

    def get_simple_side_object_torus(self):
        wheel_to_frustum = {
            2: 'top',
            3: 'bottom',
        }
        return wheel_to_frustum[self.get_wheel_count()]

    def get_simple_object_shape(self):
        load_to_b_obj = {
            "rectangle": "sphere",
            "triangle": "pyramid",
            "circle": 'cube',
            "diamond": 'cylinder',
            "hexagon": 'cone',
            "utriangle": 'torus'
        }
        return load_to_b_obj[self.l_shape]


def blender_to_michalski(blender_car_number: int, blender_color: str, blender_length: str, blender_wall: str,
                         blender_roof: str, blender_wheels: int, blender_payload_number: int,
                         blender_payload_shape: str):
    '''
    @param blender_car_number: (int) blender car number
    @param blender_color: (str) blender color, either yellow, green, grey, red, blue
    @param blender_length: (str) blender length, either long, short
    @param blender_wall: (str) blender wall, either brace_wall, solid_wall
    @param blender_roof: (str) blender roof, either roof_foundation, solid_roof, braced_roof, peaked_roof
    @param blender_wheels: (int) blender wheels, either 2, 4
    @param blender_payload_number: (int) blender payload number, either 0, 1, 2, 3
    @param blender_payload_shape: (str) blender payload shape, either diamond, box, golden_vase, barrel, metal_pot, oval_vase, none
    @return:
        n: car number (int)
        shape: car shape (str), either rectangle, bucket, ellipse, hexagon, u_shaped
        length: car length (str), either long, short
        double: car wall (str), either double, not_double
        roof: car roof (str), either none, arc, flat, jagged, peaked
        wheels: car wheels (int), either 2, 3
        l_shape: payload shape (str), either diamond, rectangle, triangle, circle, hexagon, utriangle, none
        l_num: payload number (int), either 0, 1, 2, 3
    '''
    n = blender_car_number
    shape = {
        'yellow': 'rectangle',
        'green': 'bucket',
        'grey': 'ellipse',
        'red': 'hexagon',
        'blue': 'u_shaped',
    }[blender_color]
    length = {
        'long': 'long',
        'short': 'short',
    }[blender_length]
    double = {
        'braced_wall': 'double',
        'solid_wall': 'not_double',
    }[blender_wall]
    roof = {
        'roof_foundation': 'none',
        'solid_roof': 'flat',
        'braced_roof': 'arc',
        'peaked_roof': 'peaked',
    }[blender_roof]
    wheels = {
        2: 2,
        3: 3,
    }[blender_wheels]
    l_shape = {
        'diamond': 'diamond',
        'box': 'rectangle',
        'golden_vase': 'triangle',
        'barrel': 'circle',
        'metal_pot': 'hexagon',
        'oval_vase': 'utriangle',
        'none': 'none',
    }[blender_payload_shape]
    l_num = blender_payload_number
    return n, shape, length, double, roof, wheels, l_shape, l_num

# def blender_cat():
#     color = ['yellow', 'green', 'grey', 'red', 'blue']
#     length = ['short', 'long']
#     walls = ["braced_wall", 'solid_wall']
#     roofs = ["roof_foundation", 'solid_roof', 'braced_roof', 'peaked_roof']
#     wheel_count = ['2_wheels', '3_wheels']
#     load_obj = ["box", "golden_vase", 'barrel', 'diamond', 'metal_pot', 'oval_vase']
#     return ['none'] + color + length + walls + roofs + wheel_count + load_obj
#
#
# def blender_labels():
#     return ['color', 'length', 'walls', 'roofs', 'wheel_count', 'load_obj1', 'load_obj2', 'load_obj3']
#
#
# def michalski_cat():
#     shape = ['rectangle', 'bucket', 'ellipse', 'hexagon', 'u_shaped']
#     length = ['short', 'long']
#     walls = ["double", 'not_double']
#     roofs = ["none", 'arc', 'flat', 'jagged', 'peaked']
#     wheel_count = ['2_wheels', '3_wheels']
#     load_obj = ["rectangle", "triangle", 'circle', 'diamond', 'hexagon', 'utriangle']
#     return ['none'] + shape + length + walls + roofs + wheel_count + load_obj
#
#
# def encode(li, cat):
#     return [cat.index(l) for l in li]
#
#
# def decode(li, cat):
#     return [cat[l] for l in li]
#
#
# def michalski_labels():
#     return ['shape', 'length', 'walls', 'roofs', 'wheel_count', 'load_obj1', 'load_obj2', 'load_obj3']
