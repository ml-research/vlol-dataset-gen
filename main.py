import logging
from rtpt import RTPT

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
    train_col = args.train_type
    base_scene = args.background_scene

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
    print(f'generating {train_col} images in the {base_scene}')

    # generate raw trains if they do not exist or shall be replaced
    if not os.path.isfile(f'raw/datasets/{train_col}.txt') or replace_raw:
        gen_raw_trains(train_col, with_occlusion=with_occlusion, num_entries=ds_size)

    # load trains
    trains = read_trains(f'raw/datasets/{train_col}.txt')

    # render trains
    trains = trains[start_ind:end_ind]
    rtpt = RTPT(name_initials='LH', experiment_name=f'gen_{base_scene[:3]}_{train_col[0]}',
                max_iterations=end_ind - start_ind)
    rtpt.start()
    for t_num, train in enumerate(trains, start=start_ind):
        rtpt.step()
        generate_image(base_scene, train_col, t_num, train, save_blender, replace_existing_img,
                       high_res=high_res, gen_depth=True)


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

    parser.add_argument('--train_type', type=str, default='MichalskiTrains',
                        help='whether to generate MichalskiTrains or RandomTrains')
    parser.add_argument('--background_scene', type=str, default='base_scene',
                        help='Scene in which the trains are set: base_scene, desert_scene, sky_scene or fisheye_scene')

    parser.add_argument('--cuda', type=int, default=0,
                        help='Which cuda device to use')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    main()
