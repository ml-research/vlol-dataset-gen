import math
import jsonpickle


class MichalskiTrain(object):
    '''
    A Michalski train object that contains the train data (either a list of BlenderCar objects or a list of SimpleCar objects)
    i.e. a list of MichalskiCar objects and the train label and angle of rotation as well as its scale.
    '''

    def __init__(self, m_cars, direction, angle, scale=(0.5, 0.5, 0.5)):
        '''
        @param m_cars: list of MichalskiCar objects
        @param direction: direction of train
        @param angle: angle of train rotation
        @param scale: scale of train
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
        return str(self.n) + " " + self.shape + " " + self.length + " " + self.double + " " + self.roof + " " + str(self.wheels) + " " + self.l_shape + " " + str(self.l_num)


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
        # self.payload_scale = {
        #     "golden_vase": (0.015, 0.015, 0.05),
        #     "barrel": (0.4, 0.4, 0.4),
        #     "diamond": (40, 40, 40),
        #     "metal_pot": (1.2, 1.2, 1.2),
        #     "oval_vase": (2.5, 2.5, 2.5),
        # }
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
