import math

import bpy
from mathutils import Vector

from blender_image_generator.blender_util import get_new_pos, replace_material
from blender_image_generator.get_b_box import get_b_box


def load_materials():
    """
    Loading and adding all materials to the current bpy scene
    """
    path = 'data/materials/Materials.blend'
    collection_name = 'materials'

    link = False
    relative = False
    with bpy.data.libraries.load(path, link=link, relative=relative) as (data_from, data_to):
        data_to.materials = data_from.materials[:]
    # material_collection = bpy.data.collections.new(collection_name)
    # bpy.context.scene.collection.children.link(material_collection)
    # layer_collection = bpy.context.view_layer.layer_collection.children[material_collection.name]
    # layer_collection.exclude = False
    # for obj in data_to.objects:
    #     if obj is not None:
    #         material_collection.objects.link(obj)


def load_base_scene(filepath, collection):
    """
    load a new base scene
    :param:  filepath (string)        : filepath of scene which is loaded
    :param:  collection (object)      : collection in which all objects are added
    """
    # append, set to true to keep the link to the original file
    link = False
    files = []
    # link all objects
    with bpy.data.libraries.load(filepath, link=link) as (data_from, data_to):
        data_to.objects = data_from.objects[:]

        # data_to.objects = [name for name in data_from.objects if name.startswith(obj_name)]
    # bpy.ops.wm.append(directory=filepath + "/Object/", files=files)
    my_collection = bpy.data.collections.new(collection)
    bpy.context.scene.collection.children.link(my_collection)
    layer_collection = bpy.context.view_layer.layer_collection.children[my_collection.name]
    layer_collection.exclude = False

    # link object to current scene
    for obj in data_to.objects:
        if obj is not None:
            my_collection.objects.link(obj)
            if obj.name == "Camera":
                bpy.context.scene.camera = obj


def load_engine(train_collection, location, alpha, metal_mat=None, scale=(0.5, 0.5, 0.5)):
    """
    load the train engine to the scene
    :param:  train_collection (object)      : collection in which the train is added
    :param:  location (array of int)        : location (x,y,z position) where the engine is added within the scene
    :param:  alpha (int)                    : angle of the train rotation
    :param:  metal_mat (string)             : whether to paint the metal of the engin black if 'black_metal'
    """
    filepath = f'data/shapes/train/engine/engine.blend'

    collection = 'train'
    # append, set to true to keep the link to the original file
    link = False
    my_collection = bpy.data.collections.new(collection)
    train_collection.children.link(my_collection)
    load_asset(filepath, alpha, location, my_collection, link, metal_color=metal_mat, init_obj_scale=scale)


def load_rails(train_collection, location, alpha, base_scene, scale=(0.5, 0.5, 0.5)):
    """
    load the rails to the scene, Load rails into the camera's field of view only and only load and at most to the
     boundary of the subsoil
    :param:  train_collection (object)      : collection in which the rails are added
    :param:  location (array of int)        : location (x,y,z position) where the rails are starting
    :param:  alpha (int)                    : angle of the train rotation
    :param:  base_scene (string)            : whether to paint the metal of the engin black if 'black_metal'
    :param:  scale (array of int)           : scale of the engine in x,y,z direction
    """
    filepath = f'data/shapes/train/rails/rails.blend'
    collection = 'rails'
    # append, set to true to keep the link to the original file
    # link all objects
    radius = 50 * 1/scale[0]
    rail_length = 9.015 * scale[0]
    cur_loc = location.copy()
    link = False
    alpha_to_cam = alpha % math.pi
    my_collection = bpy.data.collections.new(collection)
    train_collection.children.link(my_collection)
    load_asset(filepath, alpha, location, my_collection, link, init_obj_scale=scale)
    b_box = (1, 1, 1, 1)
    while cur_loc[0] ** 2 + cur_loc[1] ** 2 < radius ** 2 and (b_box != (0, 0, 0, 0) or base_scene == 'fisheye_scene'):
        cur_loc = get_new_pos(cur_loc, rail_length, alpha_to_cam + math.pi)
        rail = load_asset(filepath, alpha, cur_loc, my_collection, link, init_obj_scale=scale)
        b_box = get_b_box(bpy.context, rail)
    cur_loc = location.copy()
    b_box = (1, 1, 1, 1)
    while b_box != (0, 0, 0, 0):
        cur_loc = get_new_pos(cur_loc, rail_length, alpha_to_cam)
        rail = load_asset(filepath, alpha, cur_loc, my_collection, link, init_obj_scale=scale)
        b_box = get_b_box(bpy.context, rail)


def load_car(car, train_tail, train_collection, alpha):
    """
    load a train car to the scene
    :param:  car (object)                   : car which is added to the scene
    :param:  train_tail (array of int)      : the rearmost location (x,y,z position) of the previous car
    :param:  train_collection (object)      : blender collection in which the car is added
    :param:  alpha (int)                    : angle of rotation
    """
    filepath_car = f'data/shapes/train/car/{car.length}_car.blend'
    filepath_wheel = f'data/shapes/train/wheels/{car.length}_car_{car.wheels}_wheels.blend'
    collection_name = 'car' + str(car.n)

    link = False
    my_collection = bpy.data.collections.new(collection_name)
    train_collection.children.link(my_collection)

    obj = load_asset(filepath_car, alpha, train_tail, my_collection, link, material=car.get_blender_car_color(),
                     pass_index=car.get_index('car'), init_obj_scale=car.get_blender_scale())
    add_position(car, obj, 'car')
    wheels = load_asset(filepath_wheel, alpha, train_tail, my_collection, link, material=car.get_blender_car_color(),
                        pass_index=car.get_index('wheels'), init_obj_scale=car.get_blender_scale())
    add_position(car, wheels, 'wheels')

    return my_collection


def load_roof(car, car_collection, train_tail, alpha):
    """
    if car has a roof load it to the scene
    :param:  car (object)                   : car which is added to the scene
    :param:  train_tail (array of int)      : the rearmost location (x,y,z position) of the previous car
    :param:  train_collection (object)      : blender collection in which the car is added
    :param:  alpha (int)                    : angle of rotation
    """
    roof_type = car.get_blender_roof()
    link = False

    # if a car has a roof it has a full wall
    if roof_type != 'none':
        filepath = f'data/shapes/train/roof/assembly/roof_assembly_{car.length}.blend'
        obj_as = load_asset(filepath, alpha, train_tail, car_collection, link,
                            material=car.get_blender_car_color(), pass_index=car.get_index('roof'),
                            init_obj_scale=car.get_blender_scale())
        filepath = f'data/shapes/train/roof/{car.length}/{roof_type}.blend'
        obj = load_asset(filepath, alpha, train_tail, car_collection, link,
                         material=car.get_blender_car_color(), pass_index=car.get_index('roof'),
                         init_obj_scale=car.get_blender_scale())
        add_position(car, obj_as + obj, 'roof')


def load_wall(car, car_collection, train_tail, alpha, wall_type):
    """
    load the wall of the car to the scene
    :param:  car (object)                   : car which is added to the scene
    :param:  train_tail (array of int)      : the rearmost location (x,y,z position) of the previous car
    :param:  train_collection (object)      : blender collection in which the car is added
    :param:  alpha (int)                    : angle of rotation
    :param:  wall_type (string)             : wall type which is loaded
    """
    if wall_type is not None:
        filepath = f'data/shapes/train/walls/{car.length}/{wall_type}.blend'
        link = False
        objs = load_asset(filepath, alpha, train_tail, car_collection, link, material=car.get_blender_car_color(),
                          pass_index=car.get_index('wall'), init_obj_scale=car.get_blender_scale())
        add_position(car, objs, 'wall')


def load_payload(car_collection, train_tail, car, alpha):
    """
    load the payload of the car to the scene
    :param:  car (object)                   : car which is added to the scene
    :param:  train_tail (array of int)      : the rearmost location (x,y,z position) of the previous car
    :param:  train_collection (object)      : blender collection in which the car is added
    :param:  alpha (int)                    : angle of rotation
    """
    payload_num = car.get_load_number()
    if payload_num > 0:
        payload = car.get_blender_payload()
        filepath = f'data/shapes/train/load/{payload}.blend'
        # xy-distance between tail and payload
        tail_to_payload = 0.15 * car.get_blender_scale()[0]
        # distance between payloads
        d_payloads = 1 * car.get_blender_scale()[0]
        # z-distance between tail and payload (height difference)
        height_payload = 1.52 * car.get_blender_scale()[0]

        payload_pos = [train_tail[0], train_tail[1], train_tail[2] + height_payload]
        payload_pos = get_new_pos(payload_pos, -tail_to_payload, alpha)

        # append, set to true to keep the link to the original file
        link = False
        with bpy.data.libraries.load(filepath, link=link) as (data_from, data_to):
            data_to.objects = data_from.objects[:]

        for obj in data_to.objects:
            if obj is not None:
                objs = [obj] + [bpy.data.objects.new(payload + str(n), obj.data.copy()) for n in range(payload_num - 1)]
                for load_num, ob in enumerate(objs):
                    pass_index = car.get_index('payload' + str(load_num))
                    add_position(car, [obj], 'payload_' + str(load_num))

                    car_collection.objects.link(ob)
                    ob.rotation_euler = car.get_payload_rotation()
                    ob.rotation_euler[2] += alpha
                    ob.scale = car.get_payload_scale()
                    ob.location = payload_pos
                    ob.pass_index = pass_index
                    payload_pos = get_new_pos(payload_pos, -d_payloads, alpha)
                    bpy.context.view_layer.update()


def create_train(train, train_collection, train_init_cord, alpha):
    """
    create and load the train into the blender scene and the given collection
    :param:  train (object)                 : train which is added to the scene
    :param:  train_init_cord (array of int) : initial location (x,y,z position) of the train
    :param:  train_collection (object)      : blender collection in which the train is added
    :param:  alpha (int)                    : angle of rotation of the train
    """
    # train is points in x direction -> thus cars need to be added in -x direction
    # the engine is located at 2.9, first car is point at the same coordinates
    train_tail_coord = train_init_cord
    scale = train.get_blender_scale()

    for car in train.m_cars:
        car_collection = load_car(car, train_tail_coord, train_collection, alpha)
        wall_type = car.get_blender_wall()
        load_wall(car, car_collection, train_tail_coord, alpha, wall_type)
        load_roof(car, car_collection, train_tail_coord, alpha)
        train_tail_coord = get_new_pos(train_tail_coord, car.get_car_length_scalar(), alpha)
        load_payload(car_collection, train_tail_coord, car, alpha)


def load_asset(filepath, alpha, location, collection, link, material=None, metal_color=None, pass_index=0,
               init_obj_rotation=(math.radians(-.125), 0, math.radians(90)), init_obj_scale=(0.5, 0.5, 0.5)):
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
            bpy.ops.object.material_slot_remove_unused()

            if material is not None:
                material2 = 'white' if material == 'white_old' else material
                replace_material(obj, 'WOOD2', material)
                replace_material(obj, 'WOOD3', material2)
                replace_material(obj, 'Metal Scratches', material2)
                replace_material(obj, 'Wooden Planks V1', material2)
                # available metal colors: 'black_metal', 'black_metal_v2'
            if metal_color is not None:
                replace_material(obj, 'Metal05 PBR', metal_color, fit_uv=False)

    return objs


def add_position(car, objs, obj_name):
    """
    save blender location of object to the train object
    :param:  car (object)                       : car to which the locations are added
    :param:  objs (array of blender objects)    : all objects which belong to the name
    :param:  obj_name (string)                  : name of the descriptor for which we need the location
    """
    num_obj = len(objs)
    cs = Vector((0, 0, 0))
    for o in objs:
        local_bbox_center = 1 / num_obj * 0.125 * sum((Vector(b) for b in o.bound_box), Vector())
        global_bbox_center = o.matrix_world @ local_bbox_center
        cs += global_bbox_center
    car.set_blender_world_cord(obj_name, cs.to_tuple())

# example calculations
# d_payloads = 0.8
# train_tail_coord = [2.9, -1.1, -0.41]
# next_train = [0.067029, -1.1, -0.41]
# box = [0.2, -1.1, 0.774]
# box2 = [1, -1.1, 0.774]
# box3 = [1.8, -1.1, 0.774]


##################################################
# # Load the asset
# name = 'The steam locomotive ROCKET'
#
# blendPath = f'blender_model/blenderkit_data/models/the_steam_locomotive_rocket/{name}.blend'
# with bpy.data.libraries.load(blendPath, link=False, relative=True) as (data_from, data_to):
#     data_to.objects = data_from.objects[:]
#
# # Put the objects loaded in a *visible* collection
# # visible is IMPORTANT, so that modifiers are evaluated
#
# my_collection = bpy.data.collections.new('__assets__')
# bpy.context.scene.collection.children.link(my_collection)
# layer_collection = bpy.context.view_layer.layer_collection.children[my_collection.name]
# layer_collection.exclude = False
# for obj in data_to.objects:
#     my_collection.objects.link(obj)
#
# # Query vertex group values
#
# obj = bpy.context.scene.objects['Plane']
#
# depsgraph = bpy.context.evaluated_depsgraph_get()
# eo = obj.evaluated_get(depsgraph)
# mesh = eo.to_mesh(preserve_all_data_layers=True, depsgraph=depsgraph)
#
# vg = eo.vertex_groups['dist']
# vg_index = vg.index
#
#
# class NoGroup:
#     weight = 0.0
#
#
# v_weights = [0.0] * len(mesh.vertices)
# for v in mesh.vertices:
#     group = NoGroup
#     # IMPORTANT(nll) v.groups can be in any order, or incomplete...
#     for g in v.groups:
#         if g.group == vg_index:
#             group = g
#             break
#     v_weights[v.index] = group.weight
#
# # values are '1.0' if modifiers are not evaluated
# # values are '0.2828...' if modifiers are ok
# print("v_weight", v_weights)
#
# # After the evaluation, we can exclude the layer collection
#
# layer_collection.exclude = True

#############################################

# name = 'The steam locomotive ROCKET'
# filepath = f'blender_model/blenderkit_data/models/the_steam_locomotive_rocket/{name}.blend'
#
# # name of object(s) to append or link
# obj_name = 'The steam locomotive ROCKET'
#
# # append, set to true to keep the link to the original file
# link = False
# files = []
# # link all objects starting with 'Cube'
# with bpy.data.libraries.load(filepath, link=link) as (data_from, data_to):
#     for name in data_from, obj_name:
#         files.append({'name': name})
#     # data_to.objects = [name for name in data_from.objects if name.startswith(obj_name)]
# bpy.ops.wm.append(directory=filepath+"/Collection/", files=files)
# #link object to current scene
# for obj in data_to.objects:
#     if obj is not None:
#        bpy.context.collection.objects.link(obj)

#########################################

# add plane
# bpy.ops.mesh.primitive_plane_add(size=100)
# x,y,z = 0,0,0
# bpy.ops.transform.translate(value=(x, y, z))
# plane = bpy.context.active_object


# load textures
# ob = bpy.context.active_object
# mat_path = "blender_model/blenderkit_data/models/the_steam_locomotive_rocket/textures_1k"
# for img_p, c in enumerate(os.listdir(mat_path)):
#     tex = bpy.data.textures.new(name="Tex_ROCKET", type="IMAGE")
#     img = bpy.data.images.load(mat_path+img_p)
#     tex.image = img
#     tex.extension = 'EXTEND'
#     mod = ob.modifiers.new("", 'DISPLACE')
#     mod.strength = 0.1
#     mod.mid_level = 0
#     mod.texture = tex
#     ob.data.materials[c] = mat


# # create material
# mat = bpy.data.materials.new(name='Material')
# mat.use_nodes = True
# mat_nodes = mat.node_tree.nodes
# mat_links = mat.node_tree.links

# cube.data.materials.append(mat)

# metallic
# mat_nodes['Principled BSDF'].inputs['Metallic'].default_value = 1.0
# mat_nodes['Principled BSDF'].inputs['Base Color'].default_value = (
#     0.005634391214698553, 0.01852927729487419, 0.8000000715255737, 1.0)
# mat_nodes['Principled BSDF'].inputs['Roughness'].default_value = 0.167
# 0.005634391214698553
# 0.01852927729487419
# 0.8000000715255737
# 1.0
# plane.data.materials.append(mat)
