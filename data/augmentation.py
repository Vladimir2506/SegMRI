import tensorflow as tf
import numpy as np
from scipy.ndimage import zoom, rotate

class Augmentation3D(object):

    def __init__(self, crop_ratio, zoom_ratio, rotate_ratio, zoom_delta, crop_padding, rotate_angle, flip_d_ratio, flip_h_ratio, flip_w_ratio):
        
        self.crop_ratio = crop_ratio
        self.zoom_ratio = zoom_ratio
        self.rotate_ratio = rotate_ratio
        self.zoom_min = 1.0 - zoom_delta
        self.zoom_max = 1.0 + zoom_delta
        self.crop_padding = crop_padding
        self.angle_max = rotate_angle
        self.angle_min = -rotate_angle
        self.flip_d_ratio = flip_d_ratio
        self.flip_h_ratio = flip_h_ratio
        self.flip_w_ratio = flip_w_ratio

    def __call__(self, image, label):
        
        image, label = tf.py_function(self._augment_3d, [image, label], [tf.float32, tf.int32])
        return image, label

    def _augment_3d(self, image, label):

        image = image[...,0]
        label = label[...,0]
        H, W, D = image.shape
        # 0. random flip
        if np.random.random() < self.flip_d_ratio:
            image = np.flip(image, axis=2)
            label = np.flip(label, axis=2)

        if np.random.random() < self.flip_h_ratio:
            image = np.flip(image, axis=0)
            label = np.flip(label, axis=0)

        if np.random.random() < self.flip_w_ratio:
            image = np.flip(image, axis=1)
            label = np.flip(label, axis=1)

        # 1. random zoom and center crop
        if np.random.random() < self.zoom_ratio:
            zoom_factor = np.random.random() * (self.zoom_max - self.zoom_min) + self.zoom_min
            zoomed_image = zoom(image, zoom_factor, order=1)
            zoomed_label = zoom(label, zoom_factor, order=0)
            zh, zw, zd = zoomed_image.shape
            if zoom_factor <= 1.0:
                pd1 = (D - zd) // 2
                pd2 = D - zd - pd1
                ph1 = (H - zh) // 2
                ph2 = H - zh - ph1
                pw1 = (W - zw) // 2
                pw2 = W - zw - pw1
                image = np.pad(zoomed_image, ((ph1, ph2), (pw1, pw2), (pd1, pd2)))
                label = np.pad(zoomed_label, ((ph1, ph2), (pw1, pw2), (pd1, pd2)))
            else:
                zds, zhs, zws = (zd - D) // 2, (zh - H) // 2, (zw - W) // 2
                image = zoomed_image[zhs:zhs + H,zws:zws + W,zds:zds + D]
                label = zoomed_label[zhs:zhs + H,zws:zws + W,zds:zds + D]

        # 2. random pad and crop
        if np.random.random() < self.crop_ratio:
            pl = self.crop_padding
            padded_image = np.pad(image, ((pl, pl), (pl, pl), (pl, pl)))
            padded_label = np.pad(label, ((pl, pl), (pl, pl), (pl, pl)))
            cds, chs, cws = int(np.random.random() * 2 * pl), int(np.random.random() * 2 * pl), int(np.random.random() * 2 * pl)
            image = padded_image[chs:chs + H,cws:cws + W,cds:cds + D]
            label = padded_label[chs:chs + H,cws:cws + W,cds:cds + D]
        
        # 3. random rotate
        if np.random.random() < self.rotate_ratio:
            angle = np.random.random() * (self.angle_max - self.angle_min) + self.angle_min
            image = rotate(image, angle, (0, 1), False, order=1)
            label = rotate(label, angle, (0, 1), False, order=0)
            
        return image[..., np.newaxis], label[..., np.newaxis]


class Augmentation2D(object):

    def __init__(self, crop_ratio, zoom_ratio, rotate_ratio, zoom_delta, crop_padding, rotate_angle, flip_h_ratio, flip_w_ratio):
        
        self.crop_ratio = crop_ratio
        self.zoom_ratio = zoom_ratio
        self.rotate_ratio = rotate_ratio
        self.zoom_min = 1.0 - zoom_delta
        self.zoom_max = 1.0 + zoom_delta
        self.crop_padding = crop_padding
        self.angle_max = rotate_angle
        self.angle_min = -rotate_angle
        self.flip_h_ratio = flip_h_ratio
        self.flip_w_ratio = flip_w_ratio

    def __call__(self, image, label):

        image, label = tf.py_function(self._augment_2d, [image, label], [tf.float32, tf.int32])
        return image, label

    def _augment_2d(self, image, label):

        image = image[...,0]
        label = label[...,0]
        H, W, D = image.shape
        # 0. random flip
        if np.random.random() < self.flip_h_ratio:
            image = np.flip(image, axis=0)
            label = np.flip(label, axis=0)

        if np.random.random() < self.flip_w_ratio:
            image = np.flip(image, axis=1)
            label = np.flip(label, axis=1)

        # 1. random zoom and center crop
        if np.random.random() < self.zoom_ratio:
            zoom_factor = np.random.random() * (self.zoom_max - self.zoom_min) + self.zoom_min
            zoomed_image = zoom(image, zoom_factor, order=1)
            zoomed_label = zoom(label, zoom_factor, order=0)
            zh, zw = zoomed_image.shape
            if zoom_factor <= 1.0:
                ph1 = (H - zh) // 2
                ph2 = H - zh - ph1
                pw1 = (W - zw) // 2
                pw2 = W - zw - pw1
                image = np.pad(zoomed_image, ((ph1, ph2), (pw1, pw2)))
                label = np.pad(zoomed_label, ((ph1, ph2), (pw1, pw2)))
            else:
                zhs, zws = (zh - H) // 2, (zw - W) // 2
                image = zoomed_image[zhs:zhs + H,zws:zws + W]
                label = zoomed_label[zhs:zhs + H,zws:zws + W]

        # 2. random pad and crop
        if np.random.random() < self.crop_ratio:
            pl = self.crop_padding
            padded_image = np.pad(image, ((pl, pl), (pl, pl)))
            padded_label = np.pad(label, ((pl, pl), (pl, pl)))
            chs, cws = int(np.random.random() * 2 * pl), int(np.random.random() * 2 * pl)
            image = padded_image[chs:chs + H,cws:cws + W]
            label = padded_label[chs:chs + H,cws:cws + W]
        
        # 3. random rotate
        if np.random.random() < self.rotate_ratio:
            angle = np.random.random() * (self.angle_max - self.angle_min) + self.angle_min
            image = rotate(image, angle, (0, 1), False, order=1)
            label = rotate(label, angle, (0, 1), False, order=0)
            
        return image[..., np.newaxis], label[..., np.newaxis]

class AugmentationHybrid(object):

    def __init__(self, crop_ratio, zoom_ratio, rotate_ratio, zoom_delta, crop_padding, rotate_angle, flip_h_ratio, flip_w_ratio):
        
        self.crop_ratio = crop_ratio
        self.zoom_ratio = zoom_ratio
        self.rotate_ratio = rotate_ratio
        self.zoom_min = 1.0 - zoom_delta
        self.zoom_max = 1.0 + zoom_delta
        self.crop_padding = crop_padding
        self.angle_max = rotate_angle
        self.angle_min = -rotate_angle
        self.flip_h_ratio = flip_h_ratio
        self.flip_w_ratio = flip_w_ratio

    def __call__(self, image, label):
        
        image, label = tf.py_function(self._augment_h, [image, label], [tf.float32, tf.int32])
        return image, label

    def _augment_h(self, image, label):

        image = image[...,0]
        label = label[...,0]
        H, W, D = image.shape

        # 0. random flip
        if np.random.random() < self.flip_h_ratio:
            image = np.flip(image, axis=0)
            label = np.flip(label, axis=0)

        if np.random.random() < self.flip_w_ratio:
            image = np.flip(image, axis=1)
            label = np.flip(label, axis=1)

        # 1. random zoom and center crop
        if np.random.random() < self.zoom_ratio:
            zoom_factor = np.random.random() * (self.zoom_max - self.zoom_min) + self.zoom_min
            zoomed_image = []
            zoomed_label = []
            for d in range(D):
                zoomed_image.append(zoom(image[...,d], zoom_factor, order=1))
                zoomed_label.append(zoom(label[...,d], zoom_factor, order=0))
            zoomed_image = np.stack(zoomed_image, -1)
            zoomed_label = np.stack(zoomed_label, -1)
            zh, zw, _ = zoomed_image.shape
            if zoom_factor <= 1.0:
                ph1 = (H - zh) // 2
                ph2 = H - zh - ph1
                pw1 = (W - zw) // 2
                pw2 = W - zw - pw1
                image = np.pad(zoomed_image, ((ph1, ph2), (pw1, pw2), (0, 0)))
                label = np.pad(zoomed_label, ((ph1, ph2), (pw1, pw2), (0, 0)))
            else:
                zhs, zws = (zh - H) // 2, (zw - W) // 2
                image = zoomed_image[zhs:zhs + H,zws:zws + W,...]
                label = zoomed_label[zhs:zhs + H,zws:zws + W,...]

        # 2. random pad and crop
        if np.random.random() < self.crop_ratio:
            pl = self.crop_padding
            padded_image = np.pad(image, ((pl, pl), (pl, pl), (0, 0)))
            padded_label = np.pad(label, ((pl, pl), (pl, pl), (0, 0)))
            chs, cws = int(np.random.random() * 2 * pl), int(np.random.random() * 2 * pl)
            image = padded_image[chs:chs + H,cws:cws + W,...]
            label = padded_label[chs:chs + H,cws:cws + W,...]
        
        # 3. random rotate
        if np.random.random() < self.rotate_ratio:
            angle = np.random.random() * (self.angle_max - self.angle_min) + self.angle_min
            image = rotate(image, angle, (0, 1), False, order=1)
            label = rotate(label, angle, (0, 1), False, order=0)
            
        return image[..., np.newaxis], label[..., np.newaxis]

def get_augmentation(cfg_aug):

    name = cfg_aug['name']
    kwargs = cfg_aug['kwargs']
    name_ = name.lower()

    if name_ == '3d':
        return Augmentation3D(**kwargs)
    elif name_ == '2d':
        return Augmentation2D(**kwargs)
    elif name_ == 'hybrid':
        return AugmentationHybrid(**kwargs)
    else:
        raise NotImplementedError(f'[{name}] is not supported.')

