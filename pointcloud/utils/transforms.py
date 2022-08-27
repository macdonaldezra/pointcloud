import colorsys
import random
from typing import Callable, List, Optional, Tuple

import numpy as np


class DataTransformer(object):
    """
    Class to apply a set of transforms passed as parameter on class init.
    """

    def __init__(
        self,
        transforms: List[
            Callable[
                [np.ndarray, np.ndarray, np.ndarray],
                list[np.ndarray, np.ndarray, np.ndarray],
            ]
        ],
        choose: Optional[int] = None,
    ) -> None:
        self.transforms = transforms
        self.choose = choose

    def __call__(
        self, points: np.ndarray, features: np.ndarray, labels: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.choose and self.choose > 0:
            # Number of transforms to randomly select from the set of all transforms to apply
            # to the input values.
            apply_transforms = random.choice(self.transforms, k=self.choose)
        else:
            apply_transforms = self.transforms

        for transform in apply_transforms:
            points, features, labels = transform(points, features, labels)

        return (points, features, labels)


class RandomPointRotation(object):
    """
    Randomly rotate a set of points.
    """

    def __init__(self, angle: Tuple[int, int, int] = (0, 0, 1)) -> None:
        self.angle = angle

    def __call__(
        self, points: np.ndarray, features: np.ndarray, labels: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        angle_x = np.random.uniform(-self.angle[0], self.angle[0]) * np.pi
        angle_y = np.random.uniform(-self.angle[1], self.angle[1]) * np.pi
        angle_z = np.random.uniform(-self.angle[2], self.angle[2]) * np.pi
        cos_x, sin_x = np.cos(angle_x), np.sin(angle_x)
        cos_y, sin_y = np.cos(angle_y), np.sin(angle_y)
        cos_z, sin_z = np.cos(angle_z), np.sin(angle_z)
        R_x = np.array([[1, 0, 0], [0, cos_x, -sin_x], [0, sin_x, cos_x]])
        R_y = np.array([[cos_y, 0, sin_y], [0, 1, 0], [-sin_y, 0, cos_y]])
        R_z = np.array([[cos_z, -sin_z, 0], [sin_z, cos_z, 0], [0, 0, 1]])
        R = np.dot(R_z, np.dot(R_y, R_x))
        coord = np.dot(points, np.transpose(R))

        return coord, features, labels


class RandomPointScale(object):
    """
    Randomly scale pointcloud points for a given input.
    """

    def __init__(
        self, range: Tuple[float, float] = (0.8, 1.2), anisotropic: bool = False
    ) -> None:
        self.range = range
        self.anisotropic = anisotropic

    def __call__(
        self, points: np.ndarray, features: np.ndarray, labels: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.anisotropic:
            rand_scale = np.random.uniform(self.range[0], self.range[1], 3)
        else:
            rand_scale = np.random.uniform(self.range[0], self.range[1], 1)

        points *= rand_scale

        return points, features, labels


class RandomPointShift(object):
    """
    Randomly shift the uniform position of a set of pointcloud points.
    """

    def __init__(self, range: Tuple[float, float, float] = [0.2, 0.2, 0]) -> None:
        self.range = range

    def __call__(
        self, points: np.ndarray, features: np.ndarray, labels: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        x_shift = np.random.uniform(-self.range[0], self.range[0])
        y_shift = np.random.uniform(-self.range[1], self.range[1])
        z_shift = np.random.uniform(-self.range[2], self.range[2])

        points += [x_shift, y_shift, z_shift]

        return points, features, labels


class RandomPointFlip(object):
    """
    Randomly flip points along the x- or y-axis.
    """

    def __init__(self, probability: float = 0.5) -> None:
        self.probability = probability

    def __call__(
        self, points: np.ndarray, features: np.ndarray, labels: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if np.random.rand() < self.probability:
            # flip on x-axis
            points[:, 0] = -points[:, 0]
        else:
            # flip on y-axis
            points[:, 1] = -points[:, 1]

        return points, features, labels


class RandomPointJitter(object):
    """
    Randomly apply jitter to the position of a set of pointcloud points.
    """

    def __init__(self, sigma: float = 0.01, clip: float = 0.05) -> None:
        assert clip > 0, "clip input parameter must be greater than 0."

        self.sigma = sigma
        self.clip = clip

    def __call__(
        self, points: np.ndarray, features: np.ndarray, labels: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        jitter = np.clip(
            self.sigma * np.random.randn(points.shape[0], 3), -1 * self.clip, self.clip
        )
        points += jitter

        return points, features, labels


class RandomlyDropColor(object):
    """
    Randomly set all point colors to 0.
    """

    def __init__(self, probability: float = 0.15) -> None:
        self.probability = probability

    def __call__(
        self, points: np.ndarray, features: np.ndarray, labels: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if np.random.rand() < self.probability:
            features[:, :3] = 0

        return points, features, labels


class RandomlyShiftBrightness(object):
    """
    Randomly shift the brightness of the set of features.
    """

    def __init__(
        self, probability: float = 0.95, range: Tuple[float, float] = [0.4, 1.2]
    ) -> None:
        self.probability = probability
        self.range = range

    def __call__(
        self, points: np.ndarray, features: np.ndarray, labels: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if np.random.rand() < self.probability:
            features[:, :3] *= np.random.uniform(self.range[0], self.range[1])
            features[:, :3] = np.clip(features[:, :3], 0, 255)

        return points, features, labels


class ChromaticColorContrast(object):
    """
    Randomly apply chromatic auto-contrast to point colors.
    """

    def __init__(
        self, probability: float = 0.5, blend_factor: Optional[float] = None
    ) -> None:
        self.probability = probability
        self.blend_factor = blend_factor

    def __call__(
        self, points: np.ndarray, features: np.ndarray, labels: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if np.random.rand() < self.probability:
            if not self.blend_factor:
                blend_factor = np.random.rand()
            else:
                blend_factor = self.blend_factor

            min_vals = np.min(features, 0, keepdims=True)
            max_vals = np.max(features, 0, keepdims=True)

            scale = 255 / (max_vals - min_vals)

            contrast = (features[:, :3] - min_vals) * scale
            features[:, :3] = (1 - blend_factor) * features[
                :, :3
            ] + blend_factor * contrast

        return points, features, labels


class ChromaticColorTranslation(object):
    """
    Apply chromatic color translation to point colors.
    """

    def __init__(self, probability: float = 0.95, ratio: float = 0.05) -> None:
        self.probability = probability
        self.ratio = ratio

    def __call__(
        self, points: np.ndarray, features: np.ndarray, labels: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if np.random.rand() < self.probability:
            translation = (np.random.rand(1, 3) - 0.5) * 255 * 2 * self.ratio
            features[:, :3] = np.clip(translation + features[:, :3], 0, 255)

        return points, features, labels


class ChromaticColorJitter(object):
    """
    Randomly apply color jitter to point colors.
    """

    def __init__(self, probability: float = 0.95, stdev: float = 0.2) -> None:
        self.probability = probability
        self.stdev = stdev

    def __call__(
        self, points: np.ndarray, features: np.ndarray, labels: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if np.random.rand() < self.probability:
            noise = np.random.randn(features.shape[0], 3)
            noise *= self.stdev * 255
            features[:, :3] = np.clip(noise + features[:, :3], 0, 255)

        return points, features, labels


class HueSaturationTranslation(object):
    """
    Apply hue and saturation to point colors.
    """

    def __init__(self, hue_max: float = 0.5, saturation_max: float = 0.2) -> None:
        self.hue_max = hue_max
        self.saturation_max = saturation_max

    def __call__(
        self, points: np.ndarray, features: np.ndarray, labels: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        hsv_color = HueSaturationTranslation.rgb_to_hsv(features[:, :3])

        hue = (np.random.rand() - 0.5) * 2 * self.hue_max
        saturation = 1 + (np.random.rand() - 0.5) * 2 * self.saturation_max

        hsv_color[:, 0] = np.remainder(hue + hsv_color[:, 0] + 1, 1)
        hsv_color[:, 1] = np.clip(saturation * hsv_color[:, 1], 0, 1)
        features[:, :3] = np.clip(
            HueSaturationTranslation.hsv_to_rgb(hsv_color), 0, 255
        )

        return points, features, labels

    @staticmethod
    def rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
        """
        Given an nx3 numpy array of RGB colors, return the HSV color format.

        Vectorized implementation of colorsys.rgb_to_hsv retrieved from:
            https://github.com/POSTECH-CVLab/point-transformer

        Params:
            rgb: An nx3 array containing RGB colors where numbers are of type float32.

        Returns:
            An nx3 array containing HSV colors.
        """
        hsv = np.zeros_like(rgb)

        # in case an RGBA array was passed, just copy the A channel
        hsv[..., 3:] = rgb[..., 3:]
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        maxc = np.max(rgb[..., :3], axis=-1)
        minc = np.min(rgb[..., :3], axis=-1)
        hsv[..., 2] = maxc
        mask = maxc != minc
        hsv[mask, 1] = (maxc - minc)[mask] / maxc[mask]
        rc = np.zeros_like(r)
        gc = np.zeros_like(g)
        bc = np.zeros_like(b)
        rc[mask] = (maxc - r)[mask] / (maxc - minc)[mask]
        gc[mask] = (maxc - g)[mask] / (maxc - minc)[mask]
        bc[mask] = (maxc - b)[mask] / (maxc - minc)[mask]
        hsv[..., 0] = np.select(
            [r == maxc, g == maxc], [bc - gc, 2.0 + rc - bc], default=4.0 + gc - rc
        )
        hsv[..., 0] = (hsv[..., 0] / 6.0) % 1.0

        return hsv

    @staticmethod
    def hsv_to_rgb(hsv: np.ndarray):
        """
        Given an nx3 numpy array of HSV colors, return the RGB color format.

        Vectorized implementation of colorsys.hsv_to_rgb retrieved from:
            https://github.com/POSTECH-CVLab/point-transformer

        Params:
            hsv: An nx3 array containing HSV colors where numbers are of type float32.

        Returns:
            An nx3 array containing RGB colors.
        """
        rgb = np.empty_like(hsv)
        rgb[..., 3:] = hsv[..., 3:]
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        i = h * 6.0
        f = (h * 6.0) - i
        p = v * (1.0 - s)
        q = v * (1.0 - s * f)
        t = v * (1.0 - s * (1.0 - f))
        i = i % 6
        conditions = [s == 0.0, i == 1, i == 2, i == 3, i == 4, i == 5]
        rgb[..., 0] = np.select(conditions, [v, q, p, p, t, v], default=v)
        rgb[..., 1] = np.select(conditions, [v, v, v, q, p, p], default=t)
        rgb[..., 2] = np.select(conditions, [v, p, t, v, v, q], default=p)

        return rgb
