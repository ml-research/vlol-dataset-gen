import bpy


def create_tree(train, t_num, gen_depth, path_settings):
    """
    create a tree inside the compositor of blender for the given train, the tree creats of objects masks
     and depth information of the individual trains
    params:
        train (MichalskiTrain): trains for which the tree is created
        t_num (int): train id
        :param:  raw_trains (string): typ of train descriptions 'RandomTrains' or 'MichalskiTrains'
        :param:  train_vis (string): visualization of the train description either 'MichalskiTrains' or
        'SimpleObjects'
        base_scene (string): scene in which the train is placed
        gen_depth (boolean): whether to create the pixel wise depth information
    """
    # switch on nodes and get reference
    bpy.context.scene.use_nodes = True
    scene = bpy.context.scene
    tree = bpy.context.scene.node_tree
    nodes = tree.nodes
    links = tree.links
    margin = 200
    pos_x, pos_y = 300, 0
    base_path = f'output/tmp/tmp_graph_output/{path_settings}/'

    # clear default nodes
    for node in nodes:
        nodes.remove(node)

    render_layer = nodes.new(type='CompositorNodeRLayers')
    bpy.context.scene.view_layers["RenderLayer"].use_pass_object_index = True
    render_layer.location = 0, 0


    image_output = nodes.new("CompositorNodeComposite")
    # image_output.location = pos_x, pos_y + margin
    links.new(
        render_layer.outputs["Image"],
        image_output.inputs["Image"]
    )
    pos_y -= 200

    for car in train.get_cars():
        car_number = car.get_car_number()
        objs = ["wall", 'roof', 'wheels'] + [f"payload{i}" for i in range(car.get_load_number())]
        ofile_nodes = [nodes.new("CompositorNodeOutputFile") for _ in objs]
        idmask_list = [nodes.new("CompositorNodeIDMask") for _ in objs]
        math_add_nodes = [nodes.new("CompositorNodeMath") for _ in objs]

        car_id_mask = nodes.new("CompositorNodeIDMask")
        car_id_mask.location = pos_x, pos_y
        car_id_mask.index = car.get_index("car")
        link_init = car_id_mask.outputs[0]
        links.new(
            render_layer.outputs['IndexOB'],
            car_id_mask.inputs[0]
        )

        car_ofile = nodes.new("CompositorNodeOutputFile")
        car_ofile.base_path = base_path + f't_{t_num}_car_{car_number}'
        car_ofile.location = pos_x + margin * (1 + len(objs)), pos_y

        for _i, obj in enumerate(objs):
            idmask_list[_i].index = car.get_index(obj)
            idmask_list[_i].location = pos_x, pos_y + (1 + _i) * -margin

            ofile_nodes[_i].base_path = base_path + f't_{t_num}_car_{car_number}_{obj}'
            ofile_nodes[_i].location = pos_x + margin, pos_y + (1 + _i) * -margin

            math_add_nodes[_i].location = pos_x + margin * (1 + _i), pos_y
            math_add_nodes[_i].operation = "ADD"

            links.new(
                render_layer.outputs['IndexOB'],
                idmask_list[_i].inputs[0]
            )
            links.new(
                idmask_list[_i].outputs[0],
                ofile_nodes[_i].inputs['Image']
            )
            links.new(
                idmask_list[_i].outputs[0],
                math_add_nodes[_i].inputs[1]
            )
            links.new(
                link_init,
                math_add_nodes[_i].inputs[0]
            )

            link_init = math_add_nodes[_i].outputs[0]

        links.new(
            link_init,
            car_ofile.inputs['Image']
        )

        pos_y -= margin * (1 + len(objs))

    if gen_depth:
        depth_map_out = nodes.new("CompositorNodeOutputFile")
        depth_map_out.base_path = f'output/tmp/depth/{path_settings}/t_{t_num}_depth'
        depth_map_out.location = pos_x, 0
        depth_map_out.format.file_format = 'OPEN_EXR'
        links.new(
            render_layer.outputs["Depth"],
            depth_map_out.inputs["Image"]
        )

        # car_ofile = nodes.new("CompositorNodeOutputFile")
        # car_ofile.base_path = f'output/tmp/tmp_graph_output/t_{t_num}_car_{car_number}/Image0001.png'
        # car_ofile.location = 800, 285 + (car_number-1) * -800
        #
        # car_id_mask = nodes.new("CompositorNodeIDMask")
        # car_id_mask.location = 400, 285 + len(objs) * -110 + (car_number-1) * -800
        # car_id_mask.index = car.get_index("car")
        #
        # math_node = nodes.new("CompositorNodeMath")
        # math_node.location = 600, 285 + len(objs) * -110 + (car_number-1) * -800
        # math_node.operation = "ADD"
        # math_node.inputs[1].default_value = num_obj
        # links.new(
        #     render_layer.outputs['IndexOB'],
        #     math_node.inputs[0]
        # )
        #
        # for idmask in idmask_list:
        #     links.new(
        #         idmask.outputs[0],
        #         car_mask.inputs['Image']
        #     )

        # path = f'output/tmp/tmp_graph_output/t_{t_num}_car_{car_number}_wheels/Image0001.png'
        # path = f'output/tmp/tmp_graph_output/t_{t_num}_car_{car_number}_roof/Image0001.png'
        # path = f'output/tmp/tmp_graph_output/t_{t_num}_car_{car_number}_payload_{i}/Image0001.png'
        # path = f'output/tmp/tmp_graph_output/t_{t_num}_car_{car_number}/Image0001.png'

    # generate segmentation image
    # math_node = nodes.new("CompositorNodeMath")
    # math_node.location = 400, 500
    # math_node.operation = "DIVIDE"
    # math_node.inputs[1].default_value = num_obj
    # links.new(
    #     render_layer.outputs['IndexOB'],
    #     math_node.inputs[0]
    # )
    #
    # color_ramp_node = nodes.new("CompositorNodeValToRGB")
    # color_ramp_node.location = 600, 500
    # links.new(
    #     math_node.outputs["Value"],
    #     color_ramp_node.inputs[0]
    # )
    #
    # color_ramp_node.color_ramp.interpolation = "CONSTANT"
    # color_ramp_node.color_ramp.elements[0].position = 0
    # color_ramp_node.color_ramp.elements[0].color = (0, 0, 0, 1)  # black
    #
    # colors = sns.color_palette("husl", num_obj)
    #
    # [color_ramp_node.color_ramp.elements.new(i / num_obj) for i in range(1, num_obj)]
    # for i in range(num_obj):
    #     color_ramp_node.color_ramp.elements[i + 1].color = colors[i] + (1,)
    #
    # seg_output = nodes.new("CompositorNodeOutputFile")
    # seg_output.base_path = f"output/segmentation/{t_num}_m_train"
    # seg_output.location = 1000, 500
    # links.new(
    #     color_ramp_node.outputs["Image"],
    #     seg_output.inputs[0]
    # )
