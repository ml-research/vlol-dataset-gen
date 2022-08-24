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

    if args.command == 'image_generator':
        # settings
        with_occlusion = args.with_occlusion
        save_blender, high_res, gen_depth = args.save_blender, args.high_res, args.gen_depth
        replace_existing_img, replace_raw = args.replace_existing_img, args.replace_raw

        # generate images in range [start_ind:end_ind]
        ds_size = args.dataset_size
        start_ind = args.index_start
        end_ind = args.index_end if args.index_end is not None else ds_size
        if start_ind > ds_size or end_ind > ds_size:
            raise ValueError(f'start index or end index greater than dataset size')
        print(f'generating {train_vis} images using {raw_trains} descriptions in the {base_scene}')
        print(f'The images are set in the {base_scene} background')

        # generate raw trains if they do not exist or shall be replaced
        if not os.path.isfile(f'raw/datasets/{raw_trains}.txt') or replace_raw:
            gen_raw_trains(raw_trains, with_occlusion=with_occlusion, num_entries=ds_size)
        num_lines = sum(1 for line in open(f'raw/datasets/{raw_trains}.txt'))
        if num_lines != ds_size:
            raise ValueError(
                f'defined dataset size: {ds_size}\n'
                f'existing train descriptions: {num_lines}\n'
                f'{num_lines} raw train descriptions were previously generated in raw/datasets/{raw_trains}.txt \n '
                f'add \'--replace_raw\' True to command line to the replace existing train descriptions and '
                f'generate the correct number of michalski trains')
        # load trains
        trains = read_trains(f'raw/datasets/{raw_trains}.txt', toSimpleObjs=train_vis == 'SimpleObjects')

        # render trains
        trains = trains[start_ind:end_ind]
        rtpt = RTPT(name_initials='LH', experiment_name=f'gen_{base_scene[:3]}_{train_vis[0]}',
                    max_iterations=end_ind - start_ind)
        # rtpt.start()
        # for t_num, train in enumerate(trains, start=start_ind):
        #     rtpt.step()
        #     generate_image(base_scene, raw_trains, train_vis, t_num, train, save_blender, replace_existing_img,
        #                    high_res=high_res, gen_depth=gen_depth)
        # combine_json(base_scene, raw_trains, train_vis, ds_size)

    if args.command == 'vis':
        from visualization.vis import show_masked_im
        from michalski_trains import m_train_dataset
        full_ds = m_train_dataset.get_datasets(base_scene, raw_trains, train_vis, 10)
        show_masked_im(full_ds)

    if args.command == 'ct':
        from concept_tester import eval_rule
        eval_rule()

    if args.command == 'ilp':
        from popper.loop import learn_solution
        from popper.util import Settings, print_prog_score
        from ilp.setup import create_bk
        num_trains = 10
        noise = 0.0
        create_bk(num_trains, noise)
        path = 'ilp/popper/gt'
        # prog, score, stats = learn_solution(
        #     Settings(path, debug=True, show_stats=True))
        # if prog is not None:
        #     print_prog_score(prog, score)


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
    parser.add_argument('--replace_existing_img', type=bool, default=False,
                        help='If there exists already an image for the id shall it be replaced?')
    parser.add_argument('--replace_raw', type=bool, default=False,
                        help='If the train descriptions are already generated shall they be replaced?')

    parser.add_argument('--dataset_size', type=int, default=10000, help='Size of the dataset we want to create')
    parser.add_argument('--index_start', type=int, default=0, help='start rendering images at index')
    parser.add_argument('--index_end', type=int, default=None, help='stop rendering images at index')

    parser.add_argument('--description', type=str, default='MichalskiTrains',
                        help='whether to generate descriptions of MichalskiTrains, RandomTrains')
    parser.add_argument('--visualization', type=str, default='Trains', help='whether to transform the generated train '
                                                                            'description and generate 3D images of: '
                                                                            'Trains or SimpleObjects')
    parser.add_argument('--background_scene', type=str, default='base_scene',
                        help='Scene in which the trains are set: base_scene, desert_scene, sky_scene or fisheye_scene')

    parser.add_argument('--cuda', type=int, default=0,
                        help='Which cuda device to use')
    parser.add_argument('--command', type=str, default='image_generator',
                        help='whether to generate images (image_generator) or visualize generated images (vis)'
                             'or concept tester (ct)')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    main()
