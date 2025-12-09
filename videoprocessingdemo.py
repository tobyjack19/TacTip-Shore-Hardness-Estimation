import os
import cv2
import numpy as np
import pandas as pd
import sys
import shutil

from skimage import data, img_as_float
from skimage.metrics import structural_similarity as ssim
from image_transforms_toby import process_image
from tactile_image_processing.utils import save_json_obj, load_json_obj, make_dir

### setup files###

#train test split
ttsplit = 0.8

data_loc = os.path.join("Datastore", "firmness_2d")
image_folder = os.path.join(data_loc, "data", "sensor_images") #, "image_3.mp4"
new_frame_folder = os.path.join(data_loc, "frame_images")
if os.path.isdir(new_frame_folder):
    overwrite = str(input("Overwrite? [y/n]"))
    if overwrite == "n" or overwrite == "N":
        sys.exit()
    else:
        shutil.rmtree(new_frame_folder)

os.makedirs(new_frame_folder, exist_ok=False)

target_im_df_train = pd.DataFrame(
        columns=[
            "sensor_image",
            "pose_x",
            "pose_y",
            "pose_z",
            "pose_Rx",
            "pose_Ry",
            "pose_Rz",
            "speed_percentage",
            "penetrometer_hardness"
        ]
    )

target_im_df_val = pd.DataFrame(
        columns=[
            "sensor_image",
            "pose_x",
            "pose_y",
            "pose_z",
            "pose_Rx",
            "pose_Ry",
            "pose_Rz",
            "speed_percentage",
            "penetrometer_hardness"
        ]
    )

target_df_train = pd.DataFrame(
        columns=[
            "sensor_image",
            "pose_x",
            "pose_y",
            "pose_z",
            "pose_Rx",
            "pose_Ry",
            "pose_Rz",
            "speed_percentage",
            "penetrometer_hardness"
        ]
    )

target_df_val = pd.DataFrame(
        columns=[
            "sensor_image",
            "pose_x",
            "pose_y",
            "pose_z",
            "pose_Rx",
            "pose_Ry",
            "pose_Rz",
            "speed_percentage",
            "penetrometer_hardness"
        ]
    )

if os.path.isfile(os.path.join(data_loc, "train_data", 'targets.csv')):
    os.remove(os.path.join(data_loc, "train_data", 'targets.csv'))

if os.path.isfile(os.path.join(data_loc, "train_data", 'targets_images.csv')):
    os.remove(os.path.join(data_loc, "train_data", 'targets_images.csv'))

if os.path.isdir(os.path.join(data_loc, "train_data", 'processed_images')):
    shutil.rmtree(os.path.join(data_loc, "train_data", 'processed_images'))

if os.path.isfile(os.path.join(data_loc, "val_data", 'targets.csv')):
    os.remove(os.path.join(data_loc, "val_data", 'targets.csv'))

if os.path.isfile(os.path.join(data_loc, "val_data", 'targets_images.csv')):
    os.remove(os.path.join(data_loc, "val_data", 'targets_images.csv'))

if os.path.isdir(os.path.join(data_loc, "val_data", "processed_images")):
    shutil.rmtree(os.path.join(data_loc, "val_data", "processed_images"))

new_train_processed_image_folder = os.path.join(data_loc, "train_data", "processed_images")
os.makedirs(new_train_processed_image_folder, exist_ok=False)

if os.path.isdir(os.path.join(data_loc, "val_data", "processed_images")):
    shutil.rmtree(os.path.join(data_loc, "val_data", "processed_images"))

new_val_processed_image_folder = os.path.join(data_loc, "val_data", "processed_images")
os.makedirs(new_val_processed_image_folder, exist_ok=False)

proc_image_params = {
    "type": "standard_tactip", 
    "image_size": [128,128], 
    "show_tactile": True, 
    "bbox": [160, 60, 480, 380], 
    "circle_mask_radius": 155, 
    "thresh": [31,-40]
}
save_json_obj(proc_image_params, os.path.join(data_loc, 'train_data', 'processed_image_params'))
save_json_obj(proc_image_params, os.path.join(data_loc, 'val_data', 'processed_image_params'))

targets_df = pd.read_csv(os.path.join(data_loc, 'data', 'targets.csv'))

ind_train = 0
ind_val = 0

sil_num_to_pen_read = [0.455471966, 0.711380373, 1.074264535, 1.612214336, 2.459099849, 3.911566174, 6.753359756, 13.74984831, 42.68940118]

### function to sort relevant frames into train and validation folders & csvs ###

def sortimages(ssim_array,image_file,i,j,count):
    global ind_train
    global ind_val

    #index of point where arm stops pressing, (starts at 0)
    ssim_min_idx=np.where(ssim_array==ssim_array.min())
    print(ssim_min_idx[0][0]+1)

    cap = cv2.VideoCapture(image_file)
    while not cap.isOpened():
        cap = cv2.VideoCapture(image_file)
        cv2.waitKey(1000)
        print("Wait for the header")

    pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
    while True:
        flag, frame = cap.read()
        if flag:
            # The frame is ready and already captured
            if pos_frame <= ssim_min_idx[0][0]+1:
                cv2.imshow('video', frame)
                pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame = process_image(frame, bbox=proc_image_params['bbox'], dims=proc_image_params['image_size'], circle_mask_radius=proc_image_params['circle_mask_radius'], thresh=proc_image_params['thresh'])
                frame_name = f"sample_{j+1}_video_{i+1}_frame_{int(pos_frame)}.png"
                save_dir = os.path.join(new_frame_folder, frame_name)
                cv2.imwrite(save_dir,frame)

                pose_x = targets_df[targets_df['sensor_image'] == f"sample_{j+1}_video_{i+1}.mp4"]['pose_x'].values[0]
                pose_y = targets_df[targets_df['sensor_image'] == f"sample_{j+1}_video_{i+1}.mp4"]['pose_y'].values[0]
                pose_z = targets_df[targets_df['sensor_image'] == f"sample_{j+1}_video_{i+1}.mp4"]['pose_z'].values[0]
                pose_Rx = targets_df[targets_df['sensor_image'] == f"sample_{j+1}_video_{i+1}.mp4"]['pose_Rx'].values[0]
                pose_Ry = targets_df[targets_df['sensor_image'] == f"sample_{j+1}_video_{i+1}.mp4"]['pose_Ry'].values[0]
                pose_Rz = targets_df[targets_df['sensor_image'] == f"sample_{j+1}_video_{i+1}.mp4"]['pose_Rz'].values[0]
                speed_percentage = targets_df[targets_df['sensor_image'] == f"sample_{j+1}_video_{i+1}.mp4"]['speed_percentage'].values[0]

                if i < (count*ttsplit):
                    cv2.imwrite(os.path.join(new_train_processed_image_folder, frame_name) ,frame)
                    target_df_train.loc[ind_train] = np.hstack([os.path.join('..', '..', 'frame_images', frame_name), pose_x, pose_y, pose_z, pose_Rx, pose_Ry, pose_Rz, speed_percentage, sil_num_to_pen_read[j]])
                    target_im_df_train.loc[ind_train] = np.hstack([frame_name, pose_x, pose_y, pose_z, pose_Rx, pose_Ry, pose_Rz, speed_percentage, sil_num_to_pen_read[j]])
                    ind_train += 1 
                else:
                    cv2.imwrite(os.path.join(new_val_processed_image_folder, frame_name) ,frame)
                    target_df_val.loc[ind_val] = np.hstack([os.path.join('..', '..', 'frame_images', frame_name), pose_x, pose_y, pose_z, pose_Rx, pose_Ry, pose_Rz, speed_percentage, sil_num_to_pen_read[j]])
                    target_im_df_val.loc[ind_val] = np.hstack([frame_name, pose_x, pose_y, pose_z, pose_Rx, pose_Ry, pose_Rz, speed_percentage, sil_num_to_pen_read[j]])
                    ind_val += 1 
        else:
            # The next frame is not ready, so we try to read it again
            cap.set(cv2.CAP_PROP_POS_FRAMES, pos_frame-1)
            print ("frame is not ready")
            # It is better to wait for a while for the next frame to be ready
            cv2.waitKey(1000)

        if cv2.waitKey(10) == 27:
            break
        if cap.get(cv2.CAP_PROP_POS_FRAMES) == ssim_min_idx[0][0]+1:
            # If the number of captured frames is equal to the total number of frames,
            # (before arm starts moving up) we stop
            break

### MAIN ###

# folder path
dir_path = image_folder
totalcount = 0
# Iterate directory
for path in os.listdir(dir_path):
    # check if current path is a file
    if os.path.isfile(os.path.join(dir_path, path)):
        totalcount += 1
print('Total File count:', totalcount)

for j in range(9):
    if os.path.isfile(os.path.join(image_folder, f"sample_{j+1}_video_1.mp4")):

        # folder path
        dir_path = image_folder
        count = 0
        # Iterate directory
        while os.path.isfile(os.path.join(dir_path, f"sample_{j+1}_video_{count+1}.mp4")):
            count += 1
        print('File count for silicone sample:', count)

        for i in range(count):
            image_file = os.path.join(image_folder, f"sample_{j+1}_video_{i+1}.mp4")
            cap = cv2.VideoCapture(image_file)
            while not cap.isOpened():
                cap = cv2.VideoCapture(image_file)
                cv2.waitKey(1000)
                print("Wait for the header")

            pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
            while True:
                flag, frame = cap.read()
                if flag:
                    # The frame is ready and already captured
                    cv2.imshow('video', frame)
                    pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    current_image = img_as_float(frame)
                    if pos_frame == 1:
                        reference_image = current_image
                        ssim_array = np.zeros(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
                        ssim_array[int(pos_frame)-1] = ssim(reference_image, current_image, data_range=current_image.max()-current_image.min())
                    else:
                        ssim_array[int(pos_frame)-1] = ssim(reference_image, current_image, data_range=current_image.max()-current_image.min())
                    print (str(pos_frame)+" frames")
                else:
                    # The next frame is not ready, so we try to read it again
                    cap.set(cv2.CAP_PROP_POS_FRAMES, pos_frame-1)
                    print ("frame is not ready")
                    # It is better to wait for a while for the next frame to be ready
                    cv2.waitKey(1000)

                if cv2.waitKey(10) == 27:
                    break
                if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
                    # If the number of captured frames is equal to the total number of frames,
                    # we stop
                    if 0 in ssim_array[:]:
                        print("SSIM Array contains null values")
                    else:
                        print(ssim_array.min())
                        sortimages(ssim_array,image_file,i,j,count)
                    print(ssim_array)
                    break
    else:
        break

#targets_images files
target_file = os.path.join(data_loc, "train_data", "targets_images.csv")
target_im_df_train.to_csv(target_file, index=False)
target_file = os.path.join(data_loc, "val_data", "targets_images.csv")
target_im_df_val.to_csv(target_file, index=False)

#target_files
target_file = os.path.join(data_loc, "train_data", "targets.csv")
target_df_train.to_csv(target_file, index=False)
target_file = os.path.join(data_loc, "val_data", "targets.csv")
target_df_val.to_csv(target_file, index=False)