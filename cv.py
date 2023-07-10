import cv2
import numpy as np
import torch
from os import path
from config import get_config
from model import MipNeRF
import imageio
from datasets import get_dataloader
from tqdm import tqdm
from pose_utils import visualize_depth, visualize_normals, to8b
import collections


def namedtuple_map(fn, tup):
    """Apply `fn` to each element of `tup` and cast to `tup`'s namedtuple."""
    return type(tup)(*map(fn, tup))


Rays = collections.namedtuple('Rays', ('origins', 'directions', 'viewdirs', 'radii', 'lossmult', 'near', 'far'))
config = get_config()
w, h = 800 // config.factor, 800 // config.factor
focal = 1200 // config.factor
near, far = 2., 6.

arrat = [[-9.9990219e-01, 4.1922452e-03, -1.3345719e-02, -5.3798322e-02],
         [-1.3988681e-02, -2.9965907e-01, 9.5394367e-01, 3.8454704e+00],
         [-4.6566129e-10, 9.5403719e-01, 2.9968831e-01, 1.2080823e+00],
         [0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 1.0000000e+00]]
o = np.array([-5.3798322e-02, 3.8454704e+00, 1.2080823e+00])
lookat = np.array([0., 0., 0.])
frame = 0
model = MipNeRF(
    use_viewdirs=config.use_viewdirs,
    randomized=config.randomized,
    ray_shape=config.ray_shape,
    white_bkgd=config.white_bkgd,
    num_levels=config.num_levels,
    num_samples=config.num_samples,
    hidden=config.hidden,
    density_noise=config.density_noise,
    density_bias=config.density_bias,
    rgb_padding=config.rgb_padding,
    resample_padding=config.resample_padding,
    min_deg=config.min_deg,
    max_deg=config.max_deg,
    viewdirs_min_deg=config.viewdirs_min_deg,
    viewdirs_max_deg=config.viewdirs_max_deg,
    device=config.device,
)

model.load_state_dict(torch.load(config.model_weight_path))
model.eval()

pre_x, pre_y = 400., 400.


def generate_rays():
    """Computes rays using a General Pinhole Camera Model
    Assumes self.h, self.w, self.focal, and self.cam_to_world exist
    """
    global pre_x, pre_y, lookat, w, h, focal, o, lookat
    d = o - lookat
    y = np.array([0, 1., 0])
    x = np.cross(y, d)
    y = np.cross(d, x)
    d = d / np.linalg.norm(d)
    x = x / np.linalg.norm(x)
    y = y / np.linalg.norm(y)
    cam_to_world = np.array([x, y, d])

    x, y = np.meshgrid(
        np.arange(w, dtype=np.float32),  # X-Axis (columns)
        np.arange(h, dtype=np.float32),  # Y-Axis (rows)
        indexing='xy')
    camera_directions = np.stack(
        [(x - w * 0.5 + 0.5) / focal,
         -(y - h * 0.5 + 0.5) / focal,
         -np.ones_like(x)],
        axis=-1)
    # Rotate ray directions from camera frame to the world frame
    directions = ((camera_directions[None, ..., None, :] * cam_to_world).sum(
        axis=-1))  # Translate camera frame's origin to the world frame
    origins = np.broadcast_to(o, directions.shape)
    viewdirs = directions / np.linalg.norm(directions, axis=-1, keepdims=True)

    # Distance from each unit-norm direction vector to its x-axis neighbor
    dx = np.sqrt(np.sum((directions[:, :-1, :, :] - directions[:, 1:, :, :]) ** 2, -1))
    dx = np.concatenate([dx, dx[:, -2:-1, :]], 1)

    # Cut the distance in half, and then round it out so that it's
    # halfway between inscribed by / circumscribed about the pixel.
    radii = dx[..., None] * 2 / np.sqrt(12)

    ones = np.ones_like(origins[..., :1])

    rays = Rays(
        origins=origins,
        directions=directions,
        viewdirs=viewdirs,
        radii=radii,
        lossmult=ones,
        near=ones * near,
        far=ones * far)

    return namedtuple_map(lambda r: torch.tensor(r).float().reshape([-1, r.shape[-1]]), rays)


def mouse_callback(event, x, y, flags, userdata):
    global pre_x, pre_y, lookat, w, h, focal, o, lookat, frame

    if event == cv2.EVENT_MOUSEWHEEL:  # 数字2上面文字有解释
        if flags < 0 and config.factor > 1:
            config.factor -= 1
        elif flags > 0 and config.factor < 100:
            config.factor += 1
        print(config.factor)
        global pre_x, pre_y, lookat, w, h, focal
        w, h = 800 // config.factor, 800 // config.factor
        focal = 1200 // config.factor
    elif event == cv2.EVENT_MOUSEMOVE:
        dx = -x + pre_x
        dy = -y + pre_y
        pre_x, pre_y = x, y
        print(x, y)
        z_axis = o - lookat
        length = np.linalg.norm(z_axis)
        yt = np.array([0, 1, 0])
        x_axis = np.cross(yt, z_axis)
        y_axis = np.cross(z_axis, x_axis)

        z_axis = z_axis / (np.linalg.norm(z_axis) + 1e-5)
        y_axis = y_axis / (np.linalg.norm(y_axis) + 1e-5)
        x_axis = x_axis / (np.linalg.norm(x_axis) + 1e-5)
        xz_angle = np.pi * dx / w / 20
        y_angle = np.pi * dy / h / 20

        z_axis = y_axis * np.sin(y_angle) + np.cos(y_angle) * np.cos(xz_angle) * z_axis - np.cos(y_angle) * np.sin(
            xz_angle) * x_axis
        lookat = o - z_axis / np.linalg.norm(z_axis)
        # print(parameter.look_at)


cv2.namedWindow('world', cv2.WINDOW_NORMAL)
cv2.resizeWindow('world', 800, 800)

cv2.namedWindow('depth', cv2.WINDOW_NORMAL)
cv2.resizeWindow('depth', 800, 800)

cv2.setMouseCallback('world', mouse_callback)

while True:
    frame += 1
    img, dist, acc = model.render_image_cv(generate_rays(), h, w, chunks=config.chunks)
    from matplotlib import pyplot as plt

    # plt.imshow(img)
    # plt.show()
    cv2.imshow('world', img)  # 显示
    # if config.visualize_depth:
    #     cv2.imshow('depth', to8b(visualize_depth(dist, acc, near, far)))
    keyboard_input = cv2.waitKey(1)
    z_axis = o - lookat
    yt = np.array([0, 1, 0])
    x_axis = np.cross(yt, z_axis)
    z_axis = z_axis / np.linalg.norm(z_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)
    # print(keyboard_input)
    if keyboard_input & 0xFF == 119:
        o += z_axis / 4
        lookat += z_axis / 4
    elif keyboard_input & 0xFF == 115:
        o -= z_axis / 4
        lookat += z_axis / 4
    elif keyboard_input & 0xFF == 97:
        o -= x_axis / 4
        lookat -= x_axis / 4
    elif keyboard_input & 0xFF == 100:
        o += x_axis / 4
        lookat += x_axis / 4
    elif keyboard_input & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

# if __name__ == "__main__":
#     config = get_config()
#     generate_rays()
