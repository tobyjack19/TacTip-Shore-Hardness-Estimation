import numpy as np
import os
import cv2
import pandas as pd
import torch

from tactile_image_processing.image_transforms import process_image
from tactile_image_processing.image_transforms import augment_image


class ImageDataGenerator(torch.utils.data.Dataset):

    def __init__(
        self,
        data_dirs,
        csv_row_to_label,
        dims=(128, 128),
        bbox=None,
        stdiz=False,
        normlz=False,
        thresh=None,
        rshift=None,
        rzoom=None,
        brightlims=None,
        noise_var=None,
        gray=True
    ):

        # check if data dirs are lists
        assert isinstance(data_dirs, list), "data_dirs should be a list!"

        self._dims = dims
        self._bbox = bbox
        self._stdiz = stdiz
        self._normlz = normlz
        self._thresh = thresh
        self._rshift = rshift
        self._rzoom = rzoom
        self._brightlims = brightlims
        self._noise_var = noise_var
        self._gray = gray

        self._csv_row_to_label = csv_row_to_label

        # load csv file
        self._label_df = self.load_data_dirs(data_dirs)

    def load_data_dirs(self, data_dirs):

        # check if images or processed images; use for all dirs
        is_processed = os.path.isdir(os.path.join(data_dirs[0], 'processed_images'))

        # add collumn for which dir data is stored in
        df_list = []
        for data_dir in data_dirs:

            # use processed images or fall back on standard images
            if is_processed:
                image_dir = os.path.join(data_dir, 'processed_images')
                df = pd.read_csv(os.path.join(data_dir, 'targets_images.csv'))
            else: 
                image_dir = os.path.join(data_dir, 'images')
                df = pd.read_csv(os.path.join(data_dir, 'targets.csv'))

            df['image_dir'] = image_dir
            df_list.append(df)

        # concat all df
        full_df = pd.concat(df_list)

        cv2.namedWindow("Input")

        return full_df

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self._label_df)))

    def __getitem__(self, index):
        'Generate one batch of data'

        processed_frame_stack = []
        # Generate data
        row = self._label_df.iloc[index]
        image_filename = os.path.join(row['image_dir'], row['sensor_image'])
        raw_image = cv2.imread(image_filename)

        processed_image = process_image(
            raw_image,
            gray=self._gray,
            bbox=self._bbox,
            dims=self._dims,
            stdiz=self._stdiz,
            normlz=self._normlz,
            thresh=self._thresh,
        )

        processed_frame_stack.append(processed_image)
        frame_name = row['sensor_image']

        cv2.imshow("Input", processed_image)
        cv2.waitKey(1)

        spacing = 1
        ####
        for i in range(4):
            if frame_name[16].isnumeric():
                if frame_name[25].isnumeric():
                    image_filename = os.path.join(row['image_dir'], f"sample_{frame_name[7]}_video_{frame_name[15:17]}_frame_{int(frame_name[24:26])+spacing}.png")
                    frame_name = f"sample_{frame_name[7]}_video_{frame_name[15:17]}_frame_{int(frame_name[24:26])+spacing}.png"
                else:
                    image_filename = os.path.join(row['image_dir'], f"sample_{frame_name[7]}_video_{frame_name[15:17]}_frame_{int(frame_name[24])+spacing}.png")
                    frame_name = f"sample_{frame_name[7]}_video_{frame_name[15:17]}_frame_{int(frame_name[24])+spacing}.png"
            else:
                if frame_name[24].isnumeric():
                    image_filename = os.path.join(row['image_dir'], f"sample_{frame_name[7]}_video_{frame_name[15]}_frame_{int(frame_name[23:25])+spacing}.png")
                    frame_name = f"sample_{frame_name[7]}_video_{frame_name[15]}_frame_{int(frame_name[23:25])+spacing}.png"
                else:
                    image_filename = os.path.join(row['image_dir'], f"sample_{frame_name[7]}_video_{frame_name[15]}_frame_{int(frame_name[23])+spacing}.png")
                    frame_name = f"sample_{frame_name[7]}_video_{frame_name[15]}_frame_{int(frame_name[23])+spacing}.png"

            raw_image = cv2.imread(image_filename)

            #print (image_filename)
            #print (raw_image)

            processed_image = process_image(
                raw_image,
                gray=self._gray,
                bbox=self._bbox,
                dims=self._dims,
                stdiz=self._stdiz,
                normlz=self._normlz,
                thresh=self._thresh,
            )

            cv2.imshow("Input", processed_image)
            cv2.waitKey(1)

            processed_frame_stack.append(processed_image)

        processed_frame_stack = np.moveaxis(processed_frame_stack, 0, -1)
        processed_frame_stack = processed_frame_stack.reshape(*self._dims, 5)

        # applt image augmentation across frames
        processed_frame_stack = augment_image(
            processed_frame_stack,
            rshift=self._rshift,
            rzoom=self._rzoom,
            brightlims=self._brightlims,
            noise_var=self._noise_var
        )

        # put the channel into first axis because pytorch
        processed_frame_stack = np.moveaxis(processed_frame_stack, -1, 0)

        # get label
        target = self._csv_row_to_label(row)
        sample = {'inputs': processed_frame_stack, 'labels': target}

        return sample


def numpy_collate(batch):
    '''
    Batch is list of len: batch_size
    Each element is dict {images: ..., labels: ...}
    Use Collate fn to ensure they are returned as np arrays.
    '''
    # list of arrays -> stacked into array
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)

    # list of lists/tuples -> recursive on each element
    elif isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]

    # list of dicts -> recursive returned as dict with same keys
    elif isinstance(batch[0], dict):
        return {key: numpy_collate([d[key] for d in batch]) for key in batch[0]}

    # list of non array element -> list of arrays
    else:
        return np.array(batch)


def demo_image_generation(
    data_dirs,
    csv_row_to_label,
    learning_params,
    image_processing_params,
    augmentation_params
):

    # Configure dataloaders
    generator_args = {**image_processing_params, **augmentation_params}
    generator = ImageDataGenerator(
        data_dirs=data_dirs,
        csv_row_to_label=csv_row_to_label,
        **generator_args
    )

    loader = torch.utils.data.DataLoader(
        generator,
        batch_size=learning_params['batch_size'],
        shuffle=learning_params['shuffle'],
        num_workers=learning_params['n_cpu'],
        collate_fn=numpy_collate
    )

    # iterate through
    for (i_batch, sample_batched) in enumerate(loader, 0):

        # shape = (batch, n_frames, width, height)
        images = sample_batched['inputs']
        labels = sample_batched['labels']
        cv2.namedWindow("example_images")

        for i in range(images.shape[0]):
            for key, item in labels.items():
                print(key, ': ', item[i])
            print('')

            # convert image to opencv format, not pytorch
            image = np.moveaxis(images[i], 0, -1)
            cv2.imshow("example_images", image)
            k = cv2.waitKey(500)
            if k == 27:    # Esc key to stop
                exit()
