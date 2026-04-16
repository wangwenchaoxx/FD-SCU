import numpy as np
import open3d as o3d
import torch
import random
import os
import sys
import errno
import os.path as osp








def get_indices(k, size=7):
    x0, y0, z0 = 0, 0, size - k
    indeces = []
    for x in range(x0, x0 + k):
        for y in range(y0, y0 + k):
            for z in range(z0, z0 + k):
                idx = z + y * size + x * size * size
                indeces.append(idx)
    return torch.tensor(indeces, device='cuda')


def upsampling_chunk(feats, index, K_fixed=8):
    device = feats.device
    N_hr = index.shape[0]

    N_lr, C = feats.shape

    K = K_fixed

    if C % K != 0:
        C = C - (C % K)
        feats = feats[:, :C]
    chunk_size = C // K

    feats_grouped = feats.view(N_lr, K, chunk_size)

    sorted_idx, order = torch.sort(index)
    is_new = torch.ones_like(sorted_idx)
    is_new[1:] = (sorted_idx[1:] != sorted_idx[:-1])
    group_start = torch.cumsum(is_new, dim=0) - 1
    rank_sorted = torch.arange(N_hr,device=device) - group_start
    rank_sorted %= K
    rank = torch.zeros_like(rank_sorted)
    rank[order] = rank_sorted
    feats_hr = feats_grouped[index,rank]
    return feats_hr


def DCT1d_matrix(N):
    A = torch.zeros((N, N))
    for miu in range(N):
        for x in range(N):
            if miu == 0:
                E_miu = np.sqrt(1 / 2)
            else:
                E_miu = 1
            A[miu, x] = E_miu * np.sqrt(2 / N) * np.cos((2 * x + 1) * miu * np.pi / 2 / N)
    return A







def normalize(points, scale=1.0):
    centroid = np.mean(points, axis=0)
    points_new = points - centroid
    m = np.max(np.sqrt(np.sum(points_new ** 2, axis=1)))
    points_new = scale * points_new / m
    return points_new, m, centroid, scale


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return


def read_point_cloud_ply(file_path, require_normal=False):
    pcd = o3d.io.read_point_cloud(file_path)
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    if not require_normal:
        return points, colors
    else:
        normals = np.asarray(pcd.normals)
        return points, colors, normals




def draw_point_cloud(points, colors, visualize=True, save_path=''):
    points = o3d.utility.Vector3dVector(points)
    colors = o3d.utility.Vector3dVector(colors)
    pcd = o3d.geometry.PointCloud(points=points)
    pcd.colors = colors
    if visualize:
        o3d.visualization.draw_geometries([pcd])
    if save_path != '':
        o3d.io.write_point_cloud(save_path, pcd)
    return


def read_mesh_ply(file_path):
    mesh = o3d.io.read_triangle_mesh(file_path)
    points = np.asarray(mesh.vertices)
    colors = np.asarray(mesh.vertex_colors)
    faces = np.asarray(mesh.triangles)
    return points, colors, faces


def draw_mesh(points, colors, faces, visualize=True, save_path=''):
    points = o3d.utility.Vector3dVector(points)
    colors = o3d.utility.Vector3dVector(colors)
    faces = o3d.utility.Vector3iVector(faces)
    mesh = o3d.geometry.TriangleMesh(vertices=points, triangles=faces)
    mesh.vertex_colors = colors
    if visualize:
        o3d.visualization.draw_geometries([mesh])
    if save_path != '':
        o3d.io.write_triangle_mesh(save_path, mesh)
    return


def display_config(args):
    print('-------SETTINGS_________')
    for arg in vars(args):
        print("%15s: %s" % (str(arg), str(getattr(args, arg))))
    print('')


def mkdir_if_missing(directory):
    if not osp.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


class Logger(object):
    """
    Write console output to external text file.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/utils/logging.py.
    """

    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            mkdir_if_missing(os.path.dirname(fpath))
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()


def find_named_module(module, query):
    """Helper function to find a named module. Returns a `nn.Module` or `None`

    Args:
        module (nn.Module): the root module
        query (str): the module name to find

    Returns:
        nn.Module or None
    """

    return next((m for n, m in module.named_modules() if n == query), None)


def find_named_buffer(module, query):
    """Helper function to find a named buffer. Returns a `torch.Tensor` or `None`

    Args:
        module (nn.Module): the root module
        query (str): the buffer name to find

    Returns:
        torch.Tensor or None
    """
    return next((b for n, b in module.named_buffers() if n == query), None)




def _update_registered_buffer(
    module,
    buffer_name,
    state_dict_key,
    state_dict,
    policy="resize_if_empty",
    dtype=torch.int,
):
    new_size = state_dict[state_dict_key].size()
    registered_buf = find_named_buffer(module, buffer_name)

    if policy in ("resize_if_empty", "resize"):
        if registered_buf is None:
            raise RuntimeError(f'buffer "{buffer_name}" was not registered')

        if policy == "resize" or registered_buf.numel() == 0:
            registered_buf.resize_(new_size)

    elif policy == "register":
        if registered_buf is not None:
            raise RuntimeError(f'buffer "{buffer_name}" was already registered')

        module.register_buffer(buffer_name, torch.empty(new_size, dtype=dtype).fill_(0))

    else:
        raise ValueError(f'Invalid policy "{policy}"')



def update_registered_buffers(
    module,
    module_name,
    buffer_names,
    state_dict,
    policy="resize_if_empty",
    dtype=torch.int,
):
    """Update the registered buffers in a module according to the tensors sized
    in a state_dict.

    (There's no way in torch to directly load a buffer with a dynamic size)

    Args:
        module (nn.Module): the module
        module_name (str): module name in the state dict
        buffer_names (list(str)): list of the buffer names to resize in the module
        state_dict (dict): the state dict
        policy (str): Update policy, choose from
            ('resize_if_empty', 'resize', 'register')
        dtype (dtype): Type of buffer to be registered (when policy is 'register')
    """
    valid_buffer_names = [n for n, _ in module.named_buffers()]
    for buffer_name in buffer_names:
        if buffer_name not in valid_buffer_names:
            raise ValueError(f'Invalid buffer name "{buffer_name}"')

    for buffer_name in buffer_names:
        _update_registered_buffer(
            module,
            buffer_name,
            f"{module_name}.{buffer_name}",
            state_dict,
            policy,
            dtype,
        )