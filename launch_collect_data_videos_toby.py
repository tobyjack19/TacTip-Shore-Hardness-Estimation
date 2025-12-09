"""
python launch_collect_data.py -r sim -s tactip -t edge_2d
"""
import os

#from tactile_data_shear.tactile_servo_control import BASE_DATA_PATH
from collect_data_videos_toby import collect_data
from setup_targets_toby import setup_targets
from tactile_image_processing.process_data.process_image_data import process_image_data, partition_data
from tactile_image_processing.utils import make_dir

from setup_collect_data_toby import setup_collect_data
from setup_collect_data_toby import BBOX, CIRCLE_MASK_RADIUS, THRESH
from tactile_servo_control.utils.parse_args import parse_args
from tactile_servo_control.utils.setup_embodiment import setup_embodiment

def launch(args):

    output_dir = os.path.join("Datastore") #'_'.join([args.robot, args.sensor])

    sil_sample_num = input("Input silicone sample number")

    for args.task in args.tasks:
        for args.data_dir, args.sample_num in zip(args.data_dirs, args.sample_nums):

            # setup save dir
            save_dir = os.path.join(output_dir, args.task, args.data_dir) # (BASE_DATA_PATH, output_dir, args.task, args.data_dir)
            image_dir = os.path.join(save_dir, "sensor_images")
            make_dir(save_dir)
            make_dir(image_dir)

            # setup parameters
            collect_params, env_params, sensor_image_params = setup_collect_data(
                args.robot,
                args.sensor,
                args.task,
                save_dir
            )

            # setup embodiment
            robot, sensor = setup_embodiment(
                env_params,
                sensor_image_params
            )

            # setup targets to collect
            target_df = setup_targets(
                collect_params,
                sil_sample_num,
                args.sample_num,
                save_dir
            )

            # collect
            collect_data(
                robot,
                sensor,
                target_df,
                image_dir,
                collect_params
            )


def process_images(args, image_params, split=None):

    output_dir = os.path.join("Datastore") # '_'.join([args.robot, args.sensor])

    for args.task in args.tasks:
        path = os.path.join(output_dir, args.task)  # (BASE_DATA_PATH, output_dir, args.task)
        data_dirs = partition_data(path, args.data_dirs, split)
        process_image_data(path, data_dirs, image_params)


if __name__ == "__main__":

    args = parse_args(
        robot='mg400',
        sensor='tactiptoby',
        tasks=['firmness_2d_v2'],
        data_dirs=['data_s9'],
        sample_nums=[50]
    )
    launch(args)

    embodiment = '_'.join([args.robot, args.sensor])
    image_params = {
        "bbox": BBOX[embodiment],
        "circle_mask_radius": CIRCLE_MASK_RADIUS[embodiment],
        "thresh": THRESH[embodiment],
    }
    process_images(args, image_params, split=0.8)
