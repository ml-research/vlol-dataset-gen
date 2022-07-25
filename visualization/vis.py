import os
from pycocotools import mask as maskUtils
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle


def show_masked_im(train_ds):
    os.makedirs('output/test_images/', exist_ok=True)
    # for im_id in range(train_ds.__len__()):
    for im_id in range(2):
        im = train_ds.get_pil_image(im_id)
        masks = train_ds.get_mask(im_id)
        # rle = masks['car_2']['wall']['mask']
        for car_id, car_masks in masks.items():
            for obj_name, mask_dict in car_masks.items():
                if isinstance(mask_dict, dict):
                    rle = None
                    if 'mask' in mask_dict.keys():
                        rle = mask_dict['mask']
                    elif obj_name == 'mask':
                        rle = mask_dict
                    if rle is not None:
                        mask = maskUtils.decode(rle)
                        fig, ax = plt.subplots()
                        ax.set_axis_off()
                        ax.imshow(im, 'gray', interpolation='none')
                        ax.imshow(mask, 'jet', interpolation='none', alpha=0.7)
                        # plt.title(f'michalski train image with overlaid mask')
                        fig.savefig(f'output/test_images/{im_id}_train_{car_id}_car_{obj_name}.png', bbox_inches='tight', pad_inches=0, dpi=387.2)
                        plt.close()

                    # fig, ax = plt.subplots()
                    # bbox = maskUtils.toBbox(rle)
                    # rect = Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=1, edgecolor='r',
                    #                  facecolor='none')
                    # ax.add_patch(rect)
                    # ax.set_axis_off()
                    # ax.imshow(im, 'gray', interpolation='none')
                    # # plt.title(f'michalski train image with overlaid mask')
                    # fig.savefig(f'output/test_images/{im_id}boxed_train.png', bbox_inches='tight', pad_inches=0, dpi=387.2)
                    # plt.close()
