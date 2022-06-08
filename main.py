import logging
from rtpt import RTPT
from util import *



os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"


# example train all colors:
# east 1 rectangle short not_double none 2 1 triangle 2 bucket short not_double arc 2 1 circle 3 ellipse short not_double flat 2 1 diamond 4 u_shaped short not_double jagged 2 1 hexagon 5 hexagon short not_double peaked 2 1 utriangle
# example full length train:
# east 1 hexagon long not_double flat 2 1 circle 2 rectangle long not_double flat 2 1 rectangle 3 rectangle long not_double flat 2 1 circle 4 bucket long not_double flat 2 1 circle
# example trains with all attributes
# east 1 rectangle long not_double none 3 2 rectangle 2 bucket long not_double arc 3 3 diamond 3 ellipse short not_double flat 2 2 circle 4 hexagon short not_double jagged 2 2 triangle
# east 1 u_shaped long not_double peaked 2 1 hexagon 2 bucket long not_double flat 2 3 utriangle


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu" if torch.cuda.is_available() else "cpu")

    # michalski train dataset settings
    train_collections = ['RandomTrains', 'MichalskiTrains']
    train_col = train_collections[1]
    scenes = ['base_scene', 'desert_scene', 'sky_scene', 'fisheye_scene']
    base_scene = scenes[0]

    # generating images
    generate_trains = True


    if generate_trains:
        from blender_image_generator.m_train_image_generation import generate_image
        from raw.read_raw_trains import read_trains
        from raw.gen_raw_trains import gen_raw_trains
        print(f'generating {train_col} images for {base_scene}')
        with_occlusion, black = False, False
        save_blender, high_res, gen_depth = False, False, False
        replace_existing_img, replace_raw = False, False
        # generate images in range [start_ind:end_ind]
        start_ind = 000
        end_ind = 10000
        # generate raw trains if they do not exist or shall be replaced
        if not os.path.isfile(f'raw/datasets/{train_col}.txt') or replace_raw:
            gen_raw_trains(train_col)
        # load trains
        trains = read_trains(f'raw/datasets/{train_col}.txt', with_occlusion)
        trains = trains[start_ind:end_ind]
        rtpt = RTPT(name_initials='LH', experiment_name=f'gen_{base_scene[:3]}_{train_col[0]}',
                    max_iterations=end_ind - start_ind)
        rtpt.start()
        for t_num, train in enumerate(trains, start=start_ind):
            rtpt.step()
            generate_image(base_scene, train_col, t_num, train, black, save_blender, replace_existing_img,
                           high_res=high_res, gen_depth=True)

if __name__ == '__main__':
    main()
