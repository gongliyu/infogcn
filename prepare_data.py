from argparse import ArgumentParser
from pathlib import Path
import json
import copy
import pickle

from tqdm import tqdm
import numpy as np

from pathutils import list_files

# 0: 'nose'
# 1: 'left_eye'
# 2: 'right_eye'
# 3: 'left_ear'
# 4: 'right_ear'

kp_names = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle']

action_categories = ['standing', 'stand_up', 'sitting', 'sit_down', 'fall_down',
                     'sit_up', 'lying_down', 'walking']

action_category_to_index = {category: i for i, category in enumerate(action_categories)}


def is_in_bed(frames, person_id, start_frame_index, end_frame_index):
    for i in range(start_frame_index, end_frame_index):
        frame = frames[i]
        if 'objects' not in frame:
            continue
        objects = frame['objects']
        object_hash = {obj['id']: obj for obj in objects}
        if person_id not in object_hash:
            continue
        if 'in' not in object_hash[person_id]:
            return False
        if not any([object_hash[oid]['category'] == 'bed' for oid in object_hash[person_id]['in']]):
            return False
    return True


def normalize_points_with_size(xy, width, height, flip=False):
    """Normalize scale points in image with size of image to (0-1).
    xy : (frames, parts, xy) or (parts, xy)
    """
    if xy.ndim == 2:
        xy = np.expand_dims(xy, 0)
    xy[:, :, 0] /= width
    xy[:, :, 1] /= height
    if flip:
        xy[:, :, 0] = 1 - xy[:, :, 0]
    return xy


def scale_pose(xy):
    """Normalize pose points by scale with max/min value of each pose.
    xy : (frames, parts, xy) or (parts, xy)
    """
    if xy.ndim == 2:
        xy = np.expand_dims(xy, 0)
    xy_min = np.nanmin(xy, axis=1)
    xy_max = np.nanmax(xy, axis=1)
    for i in range(xy.shape[0]):
        xy[i] = ((xy[i] - xy_min[i]) / (xy_max[i] - xy_min[i] + 1e-6)) * 2 - 1
    return xy.squeeze()


def collect_poses(ovideo, window_size=1, exclude_in_bed=False):
    if 'actions' not in ovideo:
        return []
    X, Y = [], []
    for action in ovideo['actions']:
        category = action['category']
        start_i, end_i = action['start_frame_index'], action['end_frame_index']
        performer_id = action['performer_id']
        if exclude_in_bed and is_in_bed(ovideo['frames'], performer_id, start_i, end_i):
            continue
        poses = []
        imh, imw = ovideo['frames'][0]['height'], ovideo['frames'][0]['width']
        for i in range(start_i, end_i):
            if i < 0 or i >= len(ovideo['frames']):
                print(f'i={i}, start_i={start_i}, end_i={end_i}, len={len(ovideo["frames"])}')
            frame = ovideo['frames'][i]
            person = None
            if 'objects' in frame:
                for obj in frame['objects']:
                    if obj['category'] == 'person' and obj['id'] == performer_id:
                        person = obj
                        break
            if person is None:
                continue
            if 'pose' not in person:
                continue
            kpts = []
            for kp_name in kp_names:
                kpt = person['pose'][kp_name]
                # kpts.append(kpt['coordinate'] + [kpt['auto_annotation_confidence']])
                kpts.append(kpt['coordinate'])
            poses.append(kpts)
        n_poses = len(poses)
        if n_poses < 1:
            continue
        poses = np.asarray(poses)
        poses[:, :, :2] = normalize_points_with_size(poses[:, :, :2], imw, imh)
        poses[:, :, :2] = scale_pose(poses[:, :, :2])
        # poses = np.concatenate((poses, np.expand_dims((poses[:, 1, :] + poses[:, 2, :]) / 2, 1)), axis=1)
        poses = poses.tolist()
        for i in range(15, n_poses, window_size):
            X.append(poses[i - 15: i])
            Y.append(action_category_to_index[category])
    return X, Y


def collect_poses_from_dir(ovideo_dir, window_size=1, exclude_in_bed=False):
    ovideo_list = list_files(ovideo_dir, 'ovideo')
    X, Y = [], []
    for ovideo_path in tqdm(ovideo_list):
        with open(ovideo_path) as f:
            ovideo = json.load(f)
        X1, Y1 = collect_poses(ovideo, window_size, exclude_in_bed=exclude_in_bed)
        X += X1
        Y += Y1
    X = np.asarray(X)
    Y = np.asarray(Y)
    return X, Y


def main(training_set_path,
         test_set_path,
         result_data_path=Path('data'),
         window_size=1,
         exclude_in_bed=False):
    result_data_path.mkdir(exist_ok=True, parents=True)
    print(f'Collect training set ...')
    train_set = collect_poses_from_dir(training_set_path, window_size, exclude_in_bed=exclude_in_bed)
    with open(result_data_path / 'train.pkl', 'wb') as f:
        pickle.dump(train_set, f)
    print(f'Collect test set ...')
    test_set = collect_poses_from_dir(test_set_path, window_size, exclude_in_bed=exclude_in_bed)
    with open(result_data_path / 'test.pkl', 'wb') as f:
        pickle.dump(test_set, f)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--test-set-path', type=Path, nargs='+',
                        default=(Path('~/src/golden-human-perception-benchmarking-dataset/data').expanduser(),))
    parser.add_argument('--training-set-path', type=Path, nargs='+',
                        default=(Path('~/src/golden-human-perception-benchmarking-dataset/data').expanduser(),))
    parser.add_argument('--result-data-path', type=Path, default=Path('data'))
    parser.add_argument('--window-size', type=int, default=1)
    parser.add_argument('--exclude-in-bed', action='store_true')
    clargs = parser.parse_args()

    main(**vars(clargs))