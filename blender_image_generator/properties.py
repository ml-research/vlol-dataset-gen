import math

material_scale = {
    'rectangle': None,
    'black': 1,  # black
    'black_metal': 1,  # black
    'black_metal_v2': 1,  # black
    # 'blue': 1,
    'green': 1,  # dark green
    'white': 1,  # dark green
    'red': 1,  # dark green
    'dark_violet': 1,  # dark green
    'blue': 1,  # dark green
    'dark_teal': 1,  # dark green
    'dark_yellow': 1,  # dark green
    'metal': 1,
    'white': 1,
    # 'red': 1,
    'white': 4,  # white
    'copper': 1,  # white
    'white_old': 0.6,  # white
    'white_old_2': 1,  # white
    'white_old_3': 0.6,  # white
    'wood': 1,  # white
    'wood_texture': 1,  # white
    'metal_2': 1,
    'metal_3': 1,
}


init_scale = {
    "box_2": (0.23, 0.23, 0.23),
    "bowl": (0.015, 0.015, 0.05),
    "bowl_2": (0.015, 0.015, 0.05),
    "bowl_3": (0.015, 0.015, 0.05),
    "bowl_4": (0.015, 0.015, 0.05),
    "bowl_5": (0.015, 0.015, 0.05),
    "bowl_6": (0.015, 0.015, 0.05),
    "bowl_7": (0.015, 0.015, 0.05),
    "bowl_8": (0.015, 0.015, 0.05),
    "barrel": (0.4, 0.4, 0.4),
    "barrel_2": (0.2, 0.2, 0.2),
    "barrel_3": (0.12, 0.12, 0.12),
    "diamond": (40, 40, 40),
    "diamond_2": (40, 40, 40),
    "diamond_3": (40, 40, 40),
    "diamond_4": (40, 40, 40),
    "pot": (1.2, 1.2, 1.2),
    "vase": (2.5, 2.5, 2.5),
    "train": (0.5, 0.5, 0.5)
}



init_rotation = {
    "barrel": (0, 0, math.radians(90)),
    "vase": (0, 0, math.radians(90)),
    "train": (math.radians(-.125), 0, math.radians(90)),
}
# train rotation x axis = -.125


get_car_length = {
    # short train length (2,9 - 0,909) = 1,991 along x axis
    "long": 3.54121375 * init_scale["train"][0],
    # long train length (2,9 - 0.067029) = 2,832971 along x axis
    "short": 2.48875 * init_scale["train"][0],
    # engine length (2,9 - 0.067029) = 2,832971 along x axis
    "engine": 3.75 * init_scale["train"][0]
}


# original
# "diamond": 'diamond',
# "rectangle": "box",
# "triangle": "bowl",
# "circle": 'barrel',
# "hexagon": 'pot',
# "utriangle": 'vase'


# "rectangle": "box_2",
# "triangle": "barrel_3",
# "circle": 'barrel_2',



# car_shape_to_material = {
#     'rectangle': [None, None],
#     'bucket': ['dark_yellow', 'dark_yellow'],
#     'ellipse': ['dark_violet', 'dark_violet'],
#     'hexagon': ['dark_teal', 'dark_teal'],
#     'u_shaped': ['dark_blue', 'dark_blue'],
# }


# black - 3 x
# blue - 6
# copper - 4.5
# dark_green - 2 x
# metal - 1 x
# white - 7
# white - 5
# white_old - 4 x
# wood - 4.5
# wood_texture - 7
# red - 4.5



# white_old_wood


# default names
# car_shape_to_material = {
#     'rectangle': None,
#     'bucket': 'Wood Painted White',  # white
#     'ellipse': 'Black Painted Wood',  # black
#     'hexagon': 'Old Wooden Planks Painted',  # green
#     'u_shaped': 'Wooden Planks Painted'  # white
# }
# available assets:
# black_painted_wood
# black_painted_wood
# old_painted_wooden
# old_wooden_planks_painted
# wood_painted_white
# wood_panel_verze2
# wood_planks_painted_variation
# wooden_planks_painted

# # "rectangle": "box",
# "rectangle": "box_2",
# "triangle": "bowl",
# "circle": 'barrel',
# "diamond": 'diamond',
# # "diamond": 'diamond_2',
# "hexagon": 'pot',
# "utriangle": 'vase'