import os
import pickle
import numpy as np
from scipy.spatial.transform import Rotation as R


def gen_point_and_flow(n_points=1024, object_loc_range=3, object_scale=1, n_frames=2, transl_scale=5):
    """
    Generate an N-point set A, apply a shared rigid transform on these points, get an N-point set B.
    :param n_points: number of points to be generated.
    :param object_loc_range: range of the location center of points in A.
    :param object_scale: scale of the points' distribution range in A.
    :param n_frames: use (n_frames - 1) rigid transforms to generate multi-frame point cloud.
    :param transl_scale: scale of the translation in the rigid transform.
    :return:
        N-point set pc_A, N-point set pc_B, rigid transfrom rot & transl.
    """
    # Generate N-point set A
    object_loc = np.random.uniform(-object_loc_range, object_loc_range, size=(3))
    pc_A = np.random.uniform(object_loc-object_scale, object_loc+object_scale, size=(n_points, 3))

    # Generate multiple rigid transform (R, t) and multi-frame point cloud
    rot_all = []
    transl_all = []
    points = [pc_A]
    for t in range(n_frames - 1):
        rot_vec = np.random.uniform(-np.pi, np.pi, size=(3))
        rot = R.from_rotvec(rot_vec)
        rot = R.as_matrix(rot)
        transl = np.random.uniform(-transl_scale, transl_scale, size=(3, 1))
        pc_B = rot @ points[-1].T + transl
        pc_B = pc_B.T

        points.append(pc_B)
        rot_all.append(rot)
        transl_all.append(transl)

    # Accumulate
    points = np.concatenate(points, 1)
    rot = np.stack(rot_all, 0)
    transl = np.stack(transl_all, 0)

    return points, rot, transl


def gen_noise(n_points=1024, noise_center=0, noise_scale=1):
    noise = np.random.normal(noise_center, noise_scale, size=(n_points, 3))
    return noise


def gen_scene(n_objects=5, n_points_per_object=(20, 200), object_loc_range=0, object_scale=5, n_frames=2, transl_scale=5):
    points_all = []
    labels_all = []
    rigid_transforms = []

    for n in range(n_objects):
        n_points = np.random.randint(n_points_per_object[0], n_points_per_object[1]+1)

        points, rot, transl = gen_point_and_flow(n_points, object_loc_range, object_scale, n_frames, transl_scale)
        points_all.append(points)
        labels = n * np.ones(n_points)
        labels_all.append(labels)
        rigid_transforms.append((rot, transl))

    points = np.concatenate(points_all, 0)
    labels = np.concatenate(labels_all, 0)
    return points, labels, rigid_transforms


if __name__ == '__main__':
    N_SCENES = 200
    # N_POINTS_PER_SCENE = (10, 50)
    # SAVE_PATH = 'toy_data/toy200'
    N_POINTS_PER_SCENE = (20, 20)
    SAVE_PATH = 'toy_data/toy200_p20_objmix10'
    os.makedirs(SAVE_PATH, exist_ok=True)

    N_OBJECTS = 5
    OBJECT_LOC_RANGE = 0
    OBJECT_SCALE = 10

    N_FRAMES = 2
    TRANSL_SCALE = 3

    for n in range(N_SCENES):
        points, labels, rigid_transforms = gen_scene(N_OBJECTS,
                                                     N_POINTS_PER_SCENE,
                                                     OBJECT_LOC_RANGE,
                                                     OBJECT_SCALE,
                                                     N_FRAMES,
                                                     TRANSL_SCALE)

        save_file = os.path.join(SAVE_PATH, 'scene_%06d.pkl'%(n))
        with open(save_file, 'wb') as f:
            pickle.dump({
                'points': points,
                'labels': labels,
                'rigid_transforms': rigid_transforms,
            }, f, protocol=pickle.HIGHEST_PROTOCOL)