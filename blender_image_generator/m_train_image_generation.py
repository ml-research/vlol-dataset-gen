import json

from blender_image_generator.blender_util import enable_gpus, clean_up
from blender_image_generator.compositor import create_tree
from blender_image_generator.load_assets import *
from blender_image_generator.json_util import restore_img, restore_depth_map
from util import *
import time


def generate_image(base_scene, train_col, t_num, train, save_blender=False, replace_existing_img=True,
                   high_res=False, gen_depth=False):
    """ assemble a michalski train, render its corresponding image and generate ground truth information
    Args:
    :param:  base_scene (string)            : background scene of the train ('base_scene', 'desert_scene', 'sky_scene', 'fisheye_scene')
    :param:  train_col (string)             : typ of trains which are generated either 'RandomTrains' or 'MichalskiTrains'
    :param:  t_num (int)                    : id of the train
    :param:  train (train obj)              : train object which is assembled and rendered
    :param:  save_blender (bool)            : whether the blender scene shall be shaved
    :param:  replace_existing_img (bool)    : if there exists already an image for the id shall it be replaced?
    :param:  gen_depth (bool)               : whether to generate the depth information of the individual scenes
    :param:  high_res (bool)                : whether to render the images in high resolution (1920x1080) or standard resolution (480x270)
    """

    start = time.time()
    output_image = f'output/image_generator/{train_col}/{base_scene}/images/{t_num}_m_train.png'
    output_blendfile = f'output/image_generator/{train_col}/{base_scene}/blendfiles/{t_num}_m_train.blend'
    output_scene = f'output/image_generator/{train_col}/{base_scene}/scenes/{t_num}_m_train.json'
    output_depth_map = f'output/image_generator/{train_col}/{base_scene}/depths/{t_num}_m_train.png'
    if os.path.isfile(output_image) and os.path.isfile(output_scene) and (os.path.isfile(
            output_depth_map) or not gen_depth) and not replace_existing_img:
        return
    os.makedirs(f'output/image_generator/{train_col}/{base_scene}/images', exist_ok=True)
    os.makedirs(f'output/image_generator/{train_col}/{base_scene}/blendfiles', exist_ok=True)
    os.makedirs(f'output/image_generator/{train_col}/{base_scene}/scenes', exist_ok=True)
    os.makedirs(f'output/image_generator/{train_col}/{base_scene}/depths', exist_ok=True)

    # collection = 'base_scene'
    # load_base_scene(filepath, collection)
    # reset scene
    # add all base scene assets
    filepath = f'data/scenes/{base_scene}.blend'
    bpy.ops.wm.open_mainfile(filepath=filepath)

    enable_gpus("CUDA")


    # render settings
    rn_scene = bpy.context.scene
    rn_scene.render.image_settings.file_format = 'PNG'
    render_args = bpy.context.scene.render
    render_args.engine = "CYCLES"

    render_args.resolution_x, render_args.resolution_y = 1920, 1080
    # render_args.tile_x, render_args.tile_y = 256, 256
    if high_res:
        render_args.resolution_percentage = 100
    else:
        render_args.resolution_percentage = 25
    # bpy.data.worlds['World'].cycles.sample_as_light = True
    rn_scene.cycles.blur_glossy = 2.0
    rn_scene.cycles.max_bounces = 20
    rn_scene.cycles.samples = 512
    rn_scene.cycles.transparent_min_bounces = 8
    rn_scene.cycles.transparent_max_bounces = 8

    # load materials
    load_materials()

    # determine train direction and initial coordinates
    # rotate train in random direction
    # degrees between 240-280 and 60 - 120 are excluded for no occlusion
    # with occlusion extends the allowed degrees by 10 degrees for each direction

    train_dir = train.get_angle()
    alpha = math.radians(train_dir)

    # This will give ground-truth information about the scene and its objects
    scene_struct = {
        'base_scene': base_scene,
        'train_type': train_col,
        'image_index': t_num,
        'image_filename': os.path.basename(output_image),
        'blender_filename': os.path.basename(output_blendfile),
        'depth_map_filename': os.path.basename(output_depth_map),
        'angle': train_dir,
        'm_train': train.toJSON(),
        'car_masks': {},
    }

    # create blender collection for the train which is to be created
    collection = 'train'
    train_collection = bpy.data.collections.new(collection)
    bpy.context.scene.collection.children.link(train_collection)
    layer_collection = bpy.context.view_layer.layer_collection.children[train_collection.name]
    layer_collection.exclude = False

    # determine train length and the starting point as a radius distance r for the engine
    loc_length = train.get_car_length('engine')
    for car in train.m_cars:
        loc_length += car.get_car_length_scalar()
    r = - loc_length / 2
    # determine engine spawn position (which is located at the end of the engine)
    offset = (train.get_car_length('engine') + 1.2) * train.get_blender_scale()[0]
    engine_pos = r + offset

    # move rotation point away from camera
    offset = [0, -0.1]
    xd = engine_pos * math.cos(alpha) + offset[0]
    yd = engine_pos * math.sin(alpha) + offset[1]
    # load rails at scale 0.6, z = -0.176
    off_z = -0.176 * train.get_blender_scale()[0] / 0.6
    train_init_cord = [xd, yd, off_z]

    # load train engine, use mat='black_metal' for black engine metal
    mat = None
    load_engine(train_collection, train_init_cord, alpha, mat)

    load_obj_time = time.time()
    # print('time needed pre set up: ' + str(load_obj_time - start))

    # load rails at scale 0.6, z = -0.155
    off_z = -0.155 * train.get_blender_scale()[0] / 0.6
    rail_cord = offset + [off_z]
    # rail_cord = get_new_pos(rail_cord, - get_car_length['engine'], alpha)
    load_rails(train_collection, rail_cord, alpha, base_scene)

    rail_time = time.time()
    # print('time needed rails: ' + str(rail_time - load_obj_time))
    # create and load trains into blender
    create_train(train, train_collection, train_init_cord, alpha)
    asset_time = time.time()
    # print('time needed asset: ' + str(asset_time - rail_time))
    # delete duplicate materials
    clean_up()
    assets_time = time.time()
    # print('time needed load assets: ' + str(assets_time - load_obj_time))

    create_tree(train, t_num, train_col, base_scene, gen_depth)
    tree_time = time.time()

    # print('time needed tree: ' + str(tree_time - asset_time))

    rn_scene.render.filepath = output_image

    # print('time needed for compositor: ' + str(setup_end - load_obj_time))
    bpy.ops.render.render(write_still=1)
    render_time = time.time()

    # print('time needed for render: ' + str(render_time - tree_time))

    obj_mask = restore_img(train, t_num, train_col, base_scene)

    scene_struct['car_masks'] = obj_mask

    if gen_depth:
        restore_depth_map(t_num, output_depth_map, train_col, base_scene)

    if save_blender:
        bpy.ops.wm.save_as_mainfile(filepath=output_blendfile)

    with open(output_scene, 'w+') as f:
        json.dump(scene_struct, f, indent=2)
    bpy.ops.wm.read_factory_settings(use_empty=True)

    fin_time = time.time()

    # print('finish it time: ' + str(fin_time - render_time))
