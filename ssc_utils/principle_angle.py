import numpy as np

def sample_orthonormal_basis(rot, transl):
    """
    Given a rigid transform defined by (rot, transl), sample a group of orthonormal basis for
    4-D linear subspace of augmented point pairs produced by this rigid transform.
    :param rot: (N, 3, 3) array.
    :param transl: (N, 3, 1) array.
    """
    # Sample a group of 4 non-coplanar 3D points
    pc_A = np.array([[1, 0, 0],
                     [0, 1, 0],
                     [0, 0, 1],
                     [1, 1, 1]])
    pc_A = pc_A.T

    # Apply rigid transforms to produce a group of basis for 3D affine subspace
    points = [pc_A]
    for t in range(rot.shape[0]):
        pc_B = rot[t] @ points[-1] + transl[t]
        points.append(pc_B)
    points = np.concatenate(points, 0)

    # Augment into a group of basis for 4D linear subspace
    points = np.concatenate((points, np.ones((1, 4))), 0)

    # Orthonormalize
    basis, _ = np.linalg.qr(points)
    return basis


def calc_principle_angle(rot1, transl1, rot2, transl2):
    """
    Given two rigid transforms defined by (rot, transl), calculate the principle angle between subspaces
    formed by point pairs produced by these wo rigid transforms.
    """
    # Sample a group of orthonormal basis for each subspace
    basis1 = sample_orthonormal_basis(rot1, transl1)
    basis2 = sample_orthonormal_basis(rot2, transl2)

    # Calculate principle angles
    _, sigmas, _ = np.linalg.svd(basis1.T @ basis2)
    sigmas = np.clip(sigmas, -1, 1)
    thetas = np.arccos(sigmas)
    theta = np.linalg.norm(thetas)
    return theta


def calc_group_principle_angle(rigid_transforms):
    """
    Calculate the principle angle between each pair for a group of rigid transforms.
    :param rigid_transforms: A list [(rot, transl), ...].
    """
    n_subspaces = len(rigid_transforms)
    theta_all = []

    for i in range(n_subspaces-1):
        for j in range(i+1, n_subspaces):
            rot1, transl1 = rigid_transforms[i]
            rot2, transl2 = rigid_transforms[j]
            theta = calc_principle_angle(rot1, transl1, rot2, transl2)
            theta_all.append(theta)

    return theta_all


if __name__ == '__main__':
    # Test
    from scipy.spatial.transform import Rotation as R

    rot_vec = np.random.uniform(-np.pi, np.pi, size=(3))
    rot = R.from_rotvec(rot_vec)
    rot1 = R.as_matrix(rot)
    transl1 = np.random.uniform(-2, 2, size=(3, 1))
    rot1, transl1 = np.stack([rot1], 0), np.stack([transl1], 0)

    rot_vec = np.random.uniform(-np.pi, np.pi, size=(3))
    rot = R.from_rotvec(rot_vec)
    rot2 = R.as_matrix(rot)
    transl2 = np.random.uniform(-2, 2, size=(3, 1))
    rot2, transl2 = np.stack([rot2], 0), np.stack([transl2], 0)

    print (calc_principle_angle(rot1, transl1, rot2, transl2))