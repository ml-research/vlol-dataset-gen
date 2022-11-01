import math
import random
import bpy


def enable_gpus(device_type):
    """
    enable GPU in blender
    params:
        device_type (object): GPU device
    """
    bpy.context.preferences.addons["cycles"].preferences.get_devices()
    devices = bpy.context.preferences.addons["cycles"].preferences.devices
    if len(devices) < 2:
        print('Blender not detecting GPU detected')
    # print available processing units
    # for device in bpy.context.preferences.addons['cycles'].preferences.devices:
    #     print(device.name)
    bpy.context.preferences.addons["cycles"].preferences.compute_device_type = device_type
    bpy.context.scene.cycles.device = "GPU"
    return None


# create camera
def create_camera():
    """
    create camera in blender
    """
    cam_data = bpy.data.cameras.new('camera')
    cam = bpy.data.objects.new('camera', cam_data)
    cam.location = (15, -3, 6)

    cam_tracker = bpy.data.objects.new('cam_tracker', None)
    cam_tracker.location = (0, 0, 2)
    bpy.context.collection.objects.link(cam_tracker)
    constraint = cam.constraints.new(type='TRACK_TO')
    constraint.target = bpy.data.objects['cam_tracker']
    bpy.context.collection.objects.link(cam)
    bpy.context.scene.camera = cam


def clean_up():
    """
    clean up the scene
    """
    mats = bpy.data.materials
    for obj in bpy.data.objects:
        for slt in obj.material_slots:
            part = slt.name.rpartition('.')
            if part[2].isnumeric() and part[0] in mats:
                slt.material = mats.get(part[0])

    # clean up unwanted obj not necessary now
    # for obj in bpy.context.scene.objects:
    #     if obj.type in ['MESH', 'CURVE', 'SURFACE', 'META', 'FONT', 'ARMATURE', 'LATTICE', 'EMPTY', 'CAMERA', 'LAMP', 'SPEAKER']:
    #         obj.select_set(True)
    #     else:
    #         obj.select_set(False)
    # bpy.ops.object.delete()

    for block in bpy.data.meshes:
        if block.users == 0:
            bpy.data.meshes.remove(block)

    for block in bpy.data.materials:
        if block.users == 0:
            bpy.data.materials.remove(block)

    for block in bpy.data.textures:
        if block.users == 0:
            bpy.data.textures.remove(block)

    for block in bpy.data.images:
        if block.users == 0:
            bpy.data.images.remove(block)


# def rotate(point, angle_degrees, axis=(0, 1, 0)):
#     theta_degrees = angle_degrees
#     theta_radians = math.radians(theta_degrees)
#     rotated_point = np.dot(rotation_matrix(axis, theta_radians), point)
#     return rotated_point
#
#
# def rotation_matrix(axis, theta):
#     """
#     Return the rotation matrix associated with counterclockwise rotation about
#     the given axis by theta radians.
#     axis (object): GPU device
#     theta (object): GPU device
#
#     """
#     axis = np.asarray(axis)
#     axis = axis / math.sqrt(np.dot(axis, axis))
#     a = math.cos(theta / 2.0)
#     b, c, d = -axis * math.sin(theta / 2.0)
#     aa, bb, cc, dd = a * a, b * b, c * c, d * d
#     bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
#     return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
#                      [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
#                      [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])


def replace_material(object, old_material, new_material, fit_uv=True):
    """
    replace a material in blender.
    params:
        object (bpy object): object for which we are replacing the material
        old_material (string): The old material name
        new_material (string): The new material name
        fit_uv (boolean): boolean if the material shall be rescaled
    """
    # all available materials and their attributes
    material_scale = {
        'white': 4,  # white
        'white_old': 0.6,  # white
        'white_old_3': 0.6,  # white
    }
    if new_material is not None and new_material != 'yellow':
        ob_mats = object.data.materials
        nm = bpy.data.materials[new_material]
        # om = bpy.data.materials[old_material]
        # object.data.materials.append(mat)

        for i in range(0, len(ob_mats)):
            # test if the old material is in the list of used materials and replace it
            # mat_name = ob_mats[i].name
            #
            # # remove material string suffix if existent
            # if mat_name[-1].isnumeric():
            #     mat_name = mat_name[0:-4]

            if old_material in ob_mats[i].name:
                ob_mats[i] = nm

                if fit_uv:
                    # fix UV layer of material
                    scale = material_scale.get(nm.name, 1)

                    for obj in bpy.context.scene.objects:
                        obj.select_set(False)
                    bpy.context.view_layer.objects.active = object
                    bpy.ops.object.editmode_toggle()
                    bpy.ops.mesh.select_all(action='SELECT')
                    bpy.ops.uv.cube_project(cube_size=scale)
                    bpy.ops.mesh.select_all(action='DESELECT')
                    bpy.ops.object.editmode_toggle()

                    # bpy.ops.uv.smart_project()

                    # if rot:
                    #     bpy.ops.uv.select_all(action='SELECT')
                    #     bpy.ops.uv.reset()
                    #     bpy.ops.mesh.uvs_rotate()
                    #     bpy.ops.uv.select_all(action='DESELECT')
                    #     object.data.update()

                    #     me = object.data
                    #     # rotate and scale uvs
                    #     me.uv_layers.active = True
                    #     uv_layer = me.uv_layers.active
                    #     pivot = Vector((0.5, 0.5))
                    #     angle = np.radians(0)
                    #     s = material_scale[nm.name]
                    #     scale = Vector((s, s))
                    #     # aspect ratio
                    #     p = object.dimensions.y / object.dimensions.x
                    #     # rotation matrix
                    #     R = Matrix((
                    #         (np.cos(angle), np.sin(angle) / p),
                    #         (-p * np.sin(angle), np.cos(angle)),
                    #     ))
                    #     # scale matrix
                    #     S = Matrix.Diagonal(scale)
                    #
                    #     RS = np.dot(S, R)
                    #
                    #     uvs = np.empty(2 * len(me.loops))
                    #     uv_layer.data.foreach_get("uv", uvs)
                    #
                    #     # rotate and scale, translate to pivot
                    #     uvs = np.dot(
                    #         uvs.reshape((-1, 2)) - pivot,
                    #         R) + pivot
                    #     # write the new UV's back
                    #     uv_layer.data.foreach_set("uv", uvs.ravel())
                    #     me.update()

def get_new_pos(init_cord, distance, alpha):
    """
    calculate new position according to initial position angle and distance
    params:
        init_cord (array): initial coordinates
        alpha (float): angle to new position
        distance (array): distance to new position
    """
    new_pos = init_cord.copy()
    new_pos[0] = new_pos[0] + distance * math.cos(alpha)
    new_pos[1] = new_pos[1] + distance * math.sin(alpha)
    return new_pos


def render_shadeless(blender_objects, path='flat.png'):
    """
    Render a version of the scene with shading disabled and unique materials
    assigned to all objects, and return a set of all colors that should be in the
    rendered image. The image itself is written to path. This is used to ensure
    that all objects will be visible in the final rendered scene.
    params:
        blender_objects (blender object): blender object
    """
    render_args = bpy.context.scene.render

    # Cache the render args we are about to clobber
    old_filepath = render_args.filepath
    old_engine = render_args.engine
    old_use_antialiasing = render_args.use_antialiasing

    # Override some render settings to have flat shading
    render_args.filepath = path
    render_args.engine = 'BLENDER_RENDER'
    render_args.use_antialiasing = False

    # Move the lights and ground to layer 2 so they don't render
    set_layer(bpy.data.objects['Lamp_Key'], 2)
    set_layer(bpy.data.objects['Lamp_Fill'], 2)
    set_layer(bpy.data.objects['Lamp_Back'], 2)
    set_layer(bpy.data.objects['Ground'], 2)

    # Add random shadeless materials to all objects
    object_colors = set()
    old_materials = []
    for i, obj in enumerate(blender_objects):
        old_materials.append(obj.data.materials[0])
        bpy.ops.material.new()
        mat = bpy.data.materials['Material']
        mat.name = 'Material_%d' % i
        while True:
            r, g, b = [random() for _ in range(3)]
            if (r, g, b) not in object_colors: break
        object_colors.add((r, g, b))
        mat.diffuse_color = [r, g, b]
        mat.use_shadeless = True
        obj.data.materials[0] = mat

    # Render the scene
    bpy.ops.render.render(write_still=True)

    # Undo the above; first restore the materials to objects
    for mat, obj in zip(old_materials, blender_objects):
        obj.data.materials[0] = mat

    # Move the lights and ground back to layer 0
    set_layer(bpy.data.objects['Lamp_Key'], 0)
    set_layer(bpy.data.objects['Lamp_Fill'], 0)
    set_layer(bpy.data.objects['Lamp_Back'], 0)
    set_layer(bpy.data.objects['desert'], 0)

    # Set the render settings back to what they were
    render_args.filepath = old_filepath
    render_args.engine = old_engine
    render_args.use_antialiasing = old_use_antialiasing

    return object_colors


def set_layer(obj, layer_idx):
    """ Move an object to a particular layer
    params:
        obj (blender object): blender object
        layer_idx (int): layer index
    """
    # Set the target layer to True first because an object must always be on
    # at least one layer.
    obj.layers[layer_idx] = True
    for i in range(len(obj.layers)):
        obj.layers[i] = (i == layer_idx)



# def load_materials():
#     """
#     Load materials from a directory. We assume that the directory contains .blend
#     files with one material each. The file X.blend has a single NodeTree item named
#     X; this NodeTree item must have a "Color" input that accepts an RGBA value.
#     """
#     mats_dir = glob.glob('data/materials/assets/materials/*/*.blend')
#     section = 'Material'
#     sec = 'NodeTree'
#
#
#     for mat in mats_dir:
#         name = os.path.basename(mat)
#         name = os.path.splitext(name)[0]
#
#         filepath = os.path.join(mat, section, name)
#         directory = os.path.join(mat, section)
#         filename = name
#
#         bpy.ops.wm.append(
#             filepath=filepath,
#             filename=filename,
#             directory=directory)


# def load_materials():
#     c_dir = pathlib.Path().resolve()
#     path = os.path.join(c_dir, 'data/materials/assets/materials/*/*.blend')
#     directories = glob.glob(path)
#     materials = {}
#
#     for dir_mat in directories:
#         link = False
#         relative = False
#         with bpy.data.libraries.load(dir_mat, link=link, relative=relative) as (data_from, data_to):
#             data_to.materials = data_from.materials[:]
#         for mat in data_to.materials:
#             mat.use_fake_user = True
#
#             # data_to.images = data_from.images[:]
#             # for attr in dir(data_to):
#             # setattr(data_to, attr, getattr(data_from, attr))


# def load_mat(mat):
#     directories = glob.glob(f'data/materials/assets/materials' + f'/{mat}/*.blend')
#     materials = {}
#
#     for dir_mat in directories:
#         link = False
#         relative = False
#         with bpy.data.libraries.load(dir_mat, link=link, relative=relative) as (data_from, data_to):
#             for attr in dir(data_to):
#                 setattr(data_to, attr, getattr(data_from, attr))
#             # data_to.materials = data_from.materials[:]
#             # data_to.images = data_from.images[:]
#             # materials[data_to.materials[0].name] = data_to.materials[0]
#             # mat = bpy.data.materials.new(name=data_from.materials[0].name)
#     return data_to.materials[0]

# for mat in data_to.materials:
#     if mat is not None:
#         obj.data.materials.append(material)


#
# def load_materials():
#     c_dir = pathlib.Path().resolve()
#     path = os.path.join(c_dir, 'data/materials/assets/materials/*/*.blend')
#     directories = glob.glob(path)
#     materials = {}
#
#     for dir_mat in directories:
#         link = False
#         relative = False
#         with bpy.data.libraries.load(dir_mat, link=link, relative=relative) as (data_from, data_to):
#             data_to.materials = data_from.materials[:]
#             data_to.images = data_from.images[:]
#         for mat in data_to.materials:
#                 mat.use_fake_user = True
