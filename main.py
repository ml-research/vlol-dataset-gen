import warnings

import torch
from rtpt import RTPT

from blender_image_generator.blender_util import get_scale
from blender_image_generator.m_train_image_generation import generate_image
from michalski_trains.dataset import combine_json
from raw.gen_raw_trains import gen_raw_trains, read_trains
from util import *
import argparse

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"


def main():
    args = parse()
    # settings general
    ds_size, out_path, device = args.dataset_size, args.output_path, torch.device(
        f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
    tag = args.tag + '_' if args.tag != "" else args.tag

    # bk settings
    bk, replace_bk = args.bk, args.replace_bk
    rule = args.classification
    min_cars, max_cars = args.min_train_length, args.max_train_length

    # image settings
    train_vis = args.visualization
    base_scene = args.background
    occ = args.occlusion
    auto_zoom = args.auto_zoom

    # render settings
    save_blender, high_res, gen_depth = args.save_blender, args.high_res, args.depth

    # parallel settings
    replace_existing_img = not args.parallel
    start_ind = args.index_start
    end_ind = args.index_end if args.index_end is not None else ds_size

    if args.command == 'image_generator':
        # generate images in range [start_ind:end_ind]
        ds_raw_path = f'{out_path}/dataset_descriptions/{rule}/{tag}{bk}_len_{min_cars}-{max_cars}.txt'
        if start_ind > ds_size or end_ind > ds_size:
            raise ValueError(f'start index {start_ind} or end index {end_ind} greater than dataset size {ds_size}')
        if min_cars > max_cars:
            warnings.warn(f'max cars {max_cars} is smaller than min cars {min_cars}, setting max cars to {min_cars}')
            max_cars = min_cars
        print(f'generating {train_vis} images using {bk} descriptions with {min_cars} to {max_cars} cars, '
              f'the labels are derived frome the {rule} classification rule, '
              f'the images are set in the {base_scene} background')

        # generate raw trains if they do not exist or shall be replaced
        if not os.path.isfile(ds_raw_path) or replace_bk:
            gen_raw_trains(bk, rule, min_cars=min_cars, max_cars=max_cars,
                           with_occlusion=occ, num_entries=ds_size, out_path=ds_raw_path)

        num_lines = sum(1 for _ in open(ds_raw_path))
        if num_lines != ds_size:
            raise ValueError(
                f'defined dataset size: {ds_size}\n'
                f'existing train descriptions: {num_lines}\n'
                f'{num_lines} raw train descriptions were previously generated in {ds_raw_path} \n '
                f'add \'--replace_bk\' to command line arguments to the replace existing train descriptions and '
                f'generate the correct number of michalski trains')

        # get scale for train if auto zoom is enabled relative to the number of cars in the train
        # else relative to max number of cars in the dataset
        scale = get_scale(max_cars, auto_zoom)
        # load trains
        trains = read_trains(ds_raw_path, toSimpleObjs=train_vis == 'SimpleObjects', scale=scale)

        # render trains
        trains = trains[start_ind:end_ind]
        rtpt = RTPT(name_initials='LH', experiment_name=f'gen_{base_scene[:3]}_{train_vis[0]}',
                    max_iterations=end_ind - start_ind)
        rtpt.start()
        # dummy tensor so rtpt shows gpu allocation
        t = torch.Tensor([0]).to(device)
        ds_name = tag + f'{train_vis}_{rule}_{bk}_{base_scene}_len_{min_cars}-{max_cars}'

        for t_num, train in enumerate(trains, start=start_ind):
            rtpt.step()
            generate_image(rule, base_scene, bk, train_vis, t_num, train, save_blender=save_blender,
                           replace_existing_img=replace_existing_img, ds_name=ds_name, high_res=high_res,
                           gen_depth=gen_depth, min_cars=min_cars, max_cars=max_cars)
        combine_json(ds_name, out_dir=out_path, ds_size=ds_size)

    if args.command == 'ct':
        from raw.concept_tester import eval_rule
        eval_rule()


def parse():
    # Instantiate the parser
    parser = argparse.ArgumentParser(description='Blender Train Generator')
    # general settings
    parser.add_argument('--dataset_size', type=int, default=10000, help='Size of the dataset we want to create')
    parser.add_argument('--output_path', type=str, default="output/image_generator",
                        help='path to the output directory')
    parser.add_argument('--tag', type=str, default="", help='add name tag to the output directory')
    parser.add_argument('--cuda', type=int, default=0, help='Which cuda device to use')

    # Background knowledge settings
    parser.add_argument('--bk', type=str, default='MichalskiTrains',
                        help='whether to generate bk based on \'MichalskiTrains\', \'RandomTrains\'')
    parser.add_argument('--classification', type=str, default='theoryx',
                        help='the classification rule used for generating the labels of the dataset, possible options: '
                             '\'theoryx\', \'easy\', \'color\', \'numerical\', \'multi\', \'complex\', \'custom\'')
    parser.add_argument('--max_train_length', type=int, default=4, help='max number of cars a train can have')
    parser.add_argument('--min_train_length', type=int, default=2, help='min number of cars a train can have')
    parser.add_argument('--replace_bk', type=bool, default=False,
                        help='Allows multiple usages of the same bk if no bk settings are changed. '
                             'By default train descriptions are not replaced. '
                             'If new train descriptions need to be rendered set to True')

    # Visualization settings
    parser.add_argument('--visualization', type=str, default='Trains', help='whether to transform the generated train '
                                                                            'description and generate 3D images of: '
                                                                            '\'Trains\' or \'SimpleObjects\'')
    parser.add_argument('--background', type=str, default='base_scene',
                        help='Scene in which the trains are set: base_scene, desert_scene, sky_scene or fisheye_scene')
    parser.add_argument('--occlusion', type=bool, default=False,
                        help='Whether to include train angles which might lead to occlusion of the individual '
                             'train attributes.')
    parser.add_argument('--auto_zoom', type=bool, default=False,
                        help='Whether to automatically zoom in or out depending on the individual train lengths '
                             ' or fix the zoom of the camera to the max length of the train.')

    # Parallelization settings
    parser.add_argument('--index_start', type=int, default=0, help='start rendering images at index')
    parser.add_argument('--index_end', type=int, default=None, help='stop rendering images at index')
    parser.add_argument('--parallel', type=bool, default=True,
                        help='Enables parallel generation of one dataset. Recommended to clear tmp folder before. '
                             'Images generated in tmp folder from previously uncompleted runs (of the same settings) '
                             'are not deleted.')

    # rendering settings
    parser.add_argument('--save_blender', type=bool, default=False,
                        help='Whether the blender scene is saved')
    parser.add_argument('--high_res', type=bool, default=False,
                        help='whether to render the images in high resolution (1920x1080) or standard resolution '
                             '(480x270)')
    parser.add_argument('--depth', type=bool, default=False,
                        help='Whether to generate the depth information of the individual scenes')

    parser.add_argument('--command', type=str, default='image_generator',
                        help='whether to generate images \'image_generator\' or execute the concept tester \'ct\' to '
                             'check how many trains are satisfied by a specified rule')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    main()
