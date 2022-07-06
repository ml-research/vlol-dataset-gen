import bpy
import math

from blender_image_generator.blender_util import get_new_pos
from blender_image_generator.load_assets import add_position


def create_simple_scene(train, train_collection, train_init_cord, alpha):
    train_tail_coord = train_init_cord

    prev_car_long = True
    length_dict = {
        'long': 1.75,
        'short': 1.5,
    }

    for car in train.m_cars:
        distance = length_dict[car.get_car_length()]
        distance += .25 if prev_car_long else 0
        train_tail_coord = get_new_pos(train_tail_coord, distance, alpha)
        prev_car_long = True if car.get_car_length() == 'long' else False

        car_collection = create_platform(car, train_tail_coord, train_collection, alpha)
        load_objects(car_collection, train_tail_coord, car, alpha)
        load_side_obj(car_collection, train_tail_coord, car, alpha)


def create_platform(car, train_tail_coord, train_collection, alpha):
    """
    load a train car to the scene
    :param:  car (object)                   : car which is added to the scene
    :param:  train_tail (array of int)      : the rearmost location (x,y,z position) of the previous car
    :param:  train_collection (object)      : blender collection in which the car is added
    :param:  alpha (int)                    : angle of rotation
    """
    collection_name = 'car' + str(car.n)

    link = False
    my_collection = bpy.data.collections.new(collection_name)
    train_collection.children.link(my_collection)

    # platform height is represented by car length
    platform_length = car.get_car_length()

    # the michalski roof shape represents the platform shape in the simple representation
    platform_shape_dict = {
        'none': 'cube',
        'roof_foundation': 'cylinder',
        'solid_roof': 'hemisphere',
        'braced_roof': 'frustum',
        'peaked_roof': 'hexagonal_prism',
    }
    pl_shape = platform_shape_dict[car.get_blender_roof()]
    scale = [.5, .5, .5]
    scale[0] = .75 if platform_length == 'long' else .5
    scale[2] = 1 if pl_shape == 'hemisphere' else .5

    filepath_car = f'data/shapes/simple_objects/platform/{pl_shape}.blend'

    # load platform / car
    obj = load_simple_asset(filepath_car, car.get_blender_car_color(), alpha, train_tail_coord, my_collection,
                            link, pass_index=car.get_index('car'), init_obj_scale=tuple(scale))
    add_position(car, obj, 'car')
    add_position(car, obj, 'wheels')
    add_position(car, obj, 'roof')
    add_position(car, obj, 'wall')

    return my_collection


def load_objects(car_collection, train_tail, car, alpha):
    """
    load the objects and place them on the platform the scene
    :param:  car (object)                   : car which is added to the scene
    :param:  train_tail (array of int)      : the rearmost location (x,y,z position) of the previous car
    :param:  train_collection (object)      : blender collection in which the car is added
    :param:  alpha (int)                    : angle of rotation
    """
    obj_shape_dict = {
        'box': 'sphere',
        'golden_vase': 'pyramid',
        'barrel': 'cube',
        'diamond': 'cylinder',
        'metal_pot': 'cone',
        'oval_vase': 'torus',
    }
    payload_num = car.get_load_number()
    # platform_height = 1 if car.get_car_length() == 'short' else 2
    if payload_num > 0:
        payload = car.get_blender_payload()
        filepath = f'data/shapes/simple_objects/objects/{obj_shape_dict[payload]}.blend'

        # distance between beginning of platform and the object
        tail_to_payload = 0.4 if payload_num == 3 else 0.2
        # distance between payloads
        d_payloads = 0.4

        payload_pos = [train_tail[0], train_tail[1], train_tail[2] + 1]
        payload_pos = get_new_pos(payload_pos, -tail_to_payload, alpha)

        # append, set to true to keep the link to the original file
        link = False
        with bpy.data.libraries.load(filepath, link=link) as (data_from, data_to):
            data_to.objects = data_from.objects[:]
        m = bpy.data.materials['orange_glossy']

        for obj in data_to.objects:
            if obj is not None:
                objs = [obj] + [bpy.data.objects.new(obj_shape_dict[payload] + str(n), obj.data.copy()) for n in
                                range(payload_num - 1)]
                for load_num, ob in enumerate(objs):
                    pass_index = car.get_index('payload' + str(load_num))
                    add_position(car, [obj], 'payload_' + str(load_num))

                    car_collection.objects.link(ob)
                    ob.rotation_euler[2] += alpha
                    ob.location = payload_pos
                    ob.pass_index = pass_index
                    ob.scale = (.15, .15, .15)
                    ob.active_material = m
                    payload_pos = get_new_pos(payload_pos, d_payloads, alpha)
                    bpy.context.view_layer.update()


def load_side_obj(collection, train_tail, car, alpha):
    filepath_torus = f'data/shapes/simple_objects/objects/torus.blend'
    filepath_frustum = f'data/shapes/simple_objects/platform/frustum.blend'
    camera_direction = alpha - math.pi / 2 if (math.pi / 2 < alpha < 3 * math.pi / 2) else alpha + math.pi / 2
    location = get_new_pos(train_tail, 1.5, camera_direction)
    m1 = bpy.data.materials['black_metal']
    m2 = bpy.data.materials['violet']
    link = False

    if car.get_wheel_count() == 3:
        with bpy.data.libraries.load(filepath_torus, link=link) as (data_from, data_to):
            data_to.objects = data_from.objects[:]
        objs = []
        for obj in data_to.objects:
            if obj is not None:
                objs.append(obj)
                collection.objects.link(obj)
                obj.rotation_euler[2] += alpha
                obj.scale = (.25, .25, .25)
                obj.location = location
                obj.pass_index = car.get_index('car')
                bpy.context.view_layer.objects.active = obj
                obj.active_material = m1
    if car.get_car_wall() == 'double':
        with bpy.data.libraries.load(filepath_frustum, link=link) as (data_from, data_to):
            data_to.objects = data_from.objects[:]
        # link object to current scene
        objs = []
        for obj in data_to.objects:
            if obj is not None:
                objs.append(obj)
                collection.objects.link(obj)
                obj.rotation_euler[2] += alpha
                obj.scale = (.25, .25, .4)
                obj.location = location
                obj.pass_index = car.get_index('car')
                bpy.context.view_layer.objects.active = obj
                obj.active_material = m2


def load_simple_asset(filepath, material, alpha, location, collection, link, shiny=False, pass_index=0,
                      init_obj_rotation=(0, 0, 0), init_obj_scale=(1, 1, 1)):
    """
    load and add an asset to the blender scene
    :param:  filepath (string)                  : path to the asset which is loaded to the scene
    :param:  materials (object)                 : the materials to be replaced from the original asset
    :param:  alpha (int)                        : angle of rotation of the asset
    :param:  location (array of float)          : coordinates (x,y,z position) where the asset should be loaded
    :param:  collection (object)                : blender collection in which the asset is added
    :param:  link (bool)                        : whether the asset shall be linked
    :param:  metal_color (object)               : color of the assets metal, if None no adjustments are made to the
    metal
    :param:  pass_index (int)                   : pass index of the asset, which is used by the compositor to identify
    the object
    :param:  init_obj_rotation (array of float) : initial rotation (x,y,z rotation) of the asset
    :param:  init_obj_scale (array of float)    : initial scale (x,y,z scale) of the asset
    """
    color_dict = {
        'yellow': (1, 1, 0, 0.8),
        'green': (0, 1, 0, 0.8),
        'grey': (1, 1, 1, 0.8),
        'red': (1, 0, 0, 0.8),
        'blue': (0, 0, 1, 0.8),
    }

    # append, set to true to keep the link to the original file
    with bpy.data.libraries.load(filepath, link=link) as (data_from, data_to):
        data_to.objects = data_from.objects[:]
    # link object to current scene
    objs = []
    for obj in data_to.objects:
        if obj is not None:
            objs.append(obj)
            collection.objects.link(obj)
            obj.rotation_euler = init_obj_rotation
            obj.rotation_euler[2] += alpha
            obj.scale = init_obj_scale
            obj.location = location
            obj.pass_index = pass_index
            bpy.context.view_layer.objects.active = obj

            if isinstance(material, str):
                # raise ValueError('unknown material defined')
                m = bpy.data.materials[material]
                # new_material = bpy.data.materials.new(name=material+'_plain')
                # new_material.diffuse_color = color_dict[material]
                #
                # if shiny:
                #     new_material.use_nodes = True
                #     nodes = new_material.node_tree.nodes
                #     links = new_material.node_tree.links
                #
                #     for node in nodes:
                #         nodes.remove(node)
                #     glossy_node = nodes.new('ShaderNodeBsdfGlossy')
                #     output_node = nodes.new('ShaderNodeOutputMaterial')
                #     links.new(
                #         glossy_node.outputs["BSDF"],
                #         output_node.inputs["Surface"]
                #     )
                obj.active_material = m

    return objs


def load_simple_engine(train_collection, train_init_cord, alpha):
    filepath = 'data/shapes/simple_objects/train/trainv1.blend'
    collection = 'train'
    # append, set to true to keep the link to the original file
    link = False
    my_collection = bpy.data.collections.new(collection)
    train_collection.children.link(my_collection)
    load_simple_asset(filepath, None, alpha, train_init_cord, my_collection, link, init_obj_scale=(.8, .4, .4))
