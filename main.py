import logging
from rtpt import RTPT

from blender_image_generator.json_util import combine_json
from blender_image_generator.m_train_image_generation import generate_image
from raw.gen_raw_trains import gen_raw_trains, read_trains
from util import *
import argparse

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"


def main():
    args = parse()

    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu" if torch.cuda.is_available() else "cpu")

    # michalski train dataset settings
    raw_trains = args.description
    train_vis = args.visualization
    base_scene = args.background_scene
    out_path = args.output_path
    rule = args.classification_rule

    if args.command == 'image_generator':
        # settings
        with_occlusion = args.with_occlusion
        save_blender, high_res, gen_depth = args.save_blender, args.high_res, args.gen_depth
        replace_existing_img, replace_raw = not args.allow_parallel, args.replace_raw

        # generate images in range [start_ind:end_ind]
        ds_size = args.dataset_size
        start_ind = args.index_start
        end_ind = args.index_end if args.index_end is not None else ds_size
        ds_raw_path = f'output/dataset_descriptions/{raw_trains}_{rule}.txt'
        if start_ind > ds_size or end_ind > ds_size:
            raise ValueError(f'start index or end index greater than dataset size')
        print(f'generating {train_vis} images using {raw_trains} descriptions the labels are derived by {rule}')
        print(f'The images are set in the {base_scene} background')

        # generate raw trains if they do not exist or shall be replaced
        if not os.path.isfile(ds_raw_path) or replace_raw:
            gen_raw_trains(raw_trains, rule, with_occlusion=with_occlusion, num_entries=ds_size, out_path=ds_raw_path)

        num_lines = sum(1 for line in open(ds_raw_path))
        if num_lines != ds_size:
            raise ValueError(
                f'defined dataset size: {ds_size}\n'
                f'existing train descriptions: {num_lines}\n'
                f'{num_lines} raw train descriptions were previously generated in raw/datasets/{raw_trains}.txt \n '
                f'add \'--replace_raw\' to command line arguments to the replace existing train descriptions and '
                f'generate the correct number of michalski trains')
        # load trains
        trains = read_trains(ds_raw_path, toSimpleObjs=train_vis == 'SimpleObjects')

        # render trains
        trains = trains[start_ind:end_ind]
        rtpt = RTPT(name_initials='LH', experiment_name=f'gen_{base_scene[:3]}_{train_vis[0]}',
                    max_iterations=end_ind - start_ind)
        rtpt.start()
        for t_num, train in enumerate(trains, start=start_ind):
            rtpt.step()
            generate_image(rule, base_scene, raw_trains, train_vis, t_num, train, save_blender, replace_existing_img,
                           high_res=high_res, gen_depth=gen_depth)
        combine_json(base_scene, raw_trains, train_vis, rule, out_dir=out_path, ds_size=ds_size)

    if args.command == 'ct':
        from concept_tester import eval_rule
        eval_rule()


def parse():
    # Instantiate the parser
    parser = argparse.ArgumentParser(description='Blender Train Generator')
    parser.add_argument('--with_occlusion', type=bool, default=False,
                        help='Whether to include train angles which might lead to occlusion of the individual '
                             'train attributes')
    parser.add_argument('--save_blender', type=bool, default=False,
                        help='Whether the blender scene is saved')
    parser.add_argument('--high_res', type=bool, default=False,
                        help='whether to render the images in high resolution (1920x1080) or standard resolution '
                             '(480x270)')
    parser.add_argument('--gen_depth', type=bool, default=False,
                        help='Whether to generate the depth information of the individual scenes')
    parser.add_argument('--allow_parallel', type=bool, default=True,
                        help=' Enables parallel generation of one dataset. Recommended to clear tmp folder before. '
                             'Images generated in tmp folder from previously uncompleted runs are not anymore deleted.')
    parser.add_argument('--replace_raw', type=bool, default=False,
                        help='Allows multiple usages of the same train descriptions and the parallel rendering of '
                             'images of one dataset. By default train descriptions are not replaced. If new train '
                             'descriptions need to be rendered set to True')

    parser.add_argument('--dataset_size', type=int, default=10000, help='Size of the dataset we want to create')
    parser.add_argument('--index_start', type=int, default=0, help='start rendering images at index')
    parser.add_argument('--index_end', type=int, default=None, help='stop rendering images at index')
    parser.add_argument('--output_path', type=str, default="output/image_generator",
                        help='path to the output directory')

    parser.add_argument('--classification_rule', type=str, default='theoryx',
                        help='the classification rule used for generating the labels of the dataset, possible options: '
                             '\'theoryx\', \'easy\', \'color\', \'numerical\', \'multi\', \'complex\', \'custom\'')
    parser.add_argument('--description', type=str, default='MichalskiTrains',
                        help='whether to generate descriptions of \'MichalskiTrains\', \'RandomTrains\'')
    parser.add_argument('--visualization', type=str, default='Trains', help='whether to transform the generated train '
                                                                            'description and generate 3D images of: '
                                                                            '\'Trains\' or \'SimpleObjects\'')
    parser.add_argument('--background_scene', type=str, default='base_scene',
                        help='Scene in which the trains are set: base_scene, desert_scene, sky_scene or fisheye_scene')

    parser.add_argument('--cuda', type=int, default=0,
                        help='Which cuda device to use')
    parser.add_argument('--command', type=str, default='image_generator',
                        help='whether to generate images \'image_generator\' or execute the concept tester \'ct\' to '
                             'check how many trains are satisfied by a specified rule')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    main()
