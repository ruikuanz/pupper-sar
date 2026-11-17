"""Find the game on the screen in Pupper's camera picture, and where Pupper stands.

Pupper's view of the game has a thick green frame. From one camera picture:

  1. undo the fisheye: straight lines must be straight before a rectangle on a
     screen can be found. The picture is remapped to an ordinary pinhole
     camera rather than an equirectangular view, because a pinhole picture
     keeps a flat screen's edges straight;
  2. find the frame: the biggest blob of frame-green, and its four corners;
  3. warp the inside of the frame to the upright game view perception.py reads;
  4. work out where the camera is. The frame's real size is known (measure it
     on the screen with a ruler: VIEW_WIDTH_M), and four known points seen by a
     known camera pin down the camera's position (cv2.solvePnP). That is what
     lets Pupper hold its lane without drifting: it can see where it stands.

No ROS in this file: python3 tests/test.py runs it on pictures it draws itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml

from moves import Position
from perception import BORDER_HUE, CONFIG, VIEW_H, VIEW_W, hue_mask

# --- measure this on your screen ---------------------------------------------
# The width of Pupper's view INCLUDING its green frame, in metres, as it is
# shown on the screen Pupper looks at. The height follows from the game's shape.
VIEW_WIDTH_M = 0.20

# --- the pinhole picture the fisheye is remapped to ----------------------------
PINHOLE_W, PINHOLE_H = 640, 480
PINHOLE_HFOV_DEG = 110.0   # wide enough to keep the screen in view from the side lanes
PINHOLE_PITCH_DEG = 0.0    # tilt the virtual camera up (+) if the screen is above Pupper's eye level
CAMERA_PARAMS = Path(__file__).resolve().parent.parent / "pupper" / "camera_params.yaml"

# --- finding the frame ---------------------------------------------------------
FRAME_MIN_SATURATION = 80
FRAME_MIN_VALUE = 80
FRAME_MIN_AREA = 0.002     # of the picture: smaller green blobs are not the game
FRAME_ASPECT_TOLERANCE = 0.5  # how far from the game's shape a blob's box may be (it is seen at an angle)

# The frame's shape: the view plus its border, from the game's config.
_BORDER = CONFIG["view"]["border"]
_OUTER_W = CONFIG["view"]["width"] + 2 * _BORDER
_OUTER_H = CONFIG["view"]["height"] + 2 * _BORDER
FRAME_ASPECT = _OUTER_W / _OUTER_H


class Pinhole:
    """Remaps the fisheye picture (the double sphere model, pupper/camera_params.yaml) to a pinhole camera."""

    def __init__(self, width: int, height: int, params_path: Path = CAMERA_PARAMS):
        p = yaml.safe_load(params_path.read_text())["camera_params"]
        self.focal = (PINHOLE_W / 2) / np.tan(np.radians(PINHOLE_HFOV_DEG / 2))
        self.matrix = np.array([[self.focal, 0, PINHOLE_W / 2], [0, self.focal, PINHOLE_H / 2], [0, 0, 1.0]])
        # A ray through every pinhole pixel: x right, y down, z forward.
        u, v = np.meshgrid(np.arange(PINHOLE_W), np.arange(PINHOLE_H))
        x = (u - PINHOLE_W / 2) / self.focal
        y = (v - PINHOLE_H / 2) / self.focal
        z = np.ones_like(x)
        pitch = np.radians(PINHOLE_PITCH_DEG)  # looking up: rotate the rays about the x axis
        y, z = y * np.cos(pitch) - z * np.sin(pitch), y * np.sin(pitch) + z * np.cos(pitch)
        # ... and where the fisheye lens puts that ray (the double sphere projection).
        d1 = np.sqrt(x * x + y * y + z * z)
        k = p["xi"] * d1 + z
        d2 = np.sqrt(x * x + y * y + k * k)
        denominator = np.maximum(p["alpha"] * d2 + (1 - p["alpha"]) * k, 1e-9)
        # The calibration is for a 1400 x 1050 picture; scale if the camera runs at another size.
        scale = width / 1400.0
        self.map_x = ((p["fx"] * x / denominator + p["cx"]) * scale).astype(np.float32)
        self.map_y = ((p["fy"] * y / denominator + p["cy"]) * scale).astype(np.float32)

    def __call__(self, fisheye_bgr: np.ndarray) -> np.ndarray:
        return cv2.remap(fisheye_bgr, self.map_x, self.map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


def _line_through(points: np.ndarray):
    vx, vy, x0, y0 = cv2.fitLine(points.astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01).ravel()
    return np.array([x0, y0]), np.array([vx, vy])


def _intersection(a, b) -> np.ndarray | None:
    (p, d), (q, e) = a, b
    cross = d[0] * e[1] - d[1] * e[0]
    if abs(cross) < 1e-6:
        return None
    t = ((q[0] - p[0]) * e[1] - (q[1] - p[1]) * e[0]) / cross
    return p + t * d


def find_frame(bgr: np.ndarray) -> np.ndarray | None:
    """The four outer corners of the green frame, as [top-left, top-right,
    bottom-right, bottom-left] pixel coordinates, or None if it isn't in the picture."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = hue_mask(hsv, BORDER_HUE, min_saturation=FRAME_MIN_SATURATION, min_value=FRAME_MIN_VALUE).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    picture_area = bgr.shape[0] * bgr.shape[1]
    best = None
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:4]:
        if cv2.contourArea(contour) < FRAME_MIN_AREA * picture_area:
            break
        quad = cv2.approxPolyDP(contour, 0.03 * cv2.arcLength(contour, True), True)
        if len(quad) != 4 or not cv2.isContourConvex(quad):
            continue
        _, _, w, h = cv2.boundingRect(quad)
        if abs(np.log((w / h) / FRAME_ASPECT)) > FRAME_ASPECT_TOLERANCE:
            continue
        best = (contour.reshape(-1, 2).astype(np.float64), quad.reshape(4, 2).astype(np.float64))
        break
    if best is None:
        return None
    points, corners = best

    # Order the corners: top-left has the smallest x + y, bottom-right the
    # largest; top-right the smallest y - x, bottom-left the largest.
    total, difference = corners.sum(axis=1), corners[:, 1] - corners[:, 0]
    corners = np.array([corners[total.argmin()], corners[difference.argmin()],
                        corners[total.argmax()], corners[difference.argmax()]])

    # approxPolyDP's corners are whole pixels picked off the outline. Through
    # the fisheye the frame is small, so every pixel counts: fit a line to the
    # outline points along each side, and put each corner where two lines cross.
    sides = []
    for i in range(4):
        a, b = corners[i], corners[(i + 1) % 4]
        along = b - a
        length = np.linalg.norm(along)
        along /= length
        offset = points - a
        t = offset @ along
        distance = np.abs(offset[:, 0] * along[1] - offset[:, 1] * along[0])
        on_side = points[(distance < 2.5) & (t > 0.15 * length) & (t < 0.85 * length)]
        if len(on_side) < 4:
            return corners.astype(np.float32)
        sides.append(_line_through(on_side))
    refined = []
    for i in range(4):
        corner = _intersection(sides[i - 1], sides[i])
        if corner is None or np.linalg.norm(corner - corners[i]) > 6.0:
            corner = corners[i]
        refined.append(corner)
    return np.array(refined, dtype=np.float32)


def game_view(bgr: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """The inside of the frame, upright, at the size perception.py reads."""
    border = _BORDER * VIEW_W / CONFIG["view"]["width"]  # the frame's thickness at that size
    w, h = VIEW_W + 2 * border, VIEW_H + 2 * border
    # The outline sits half a pixel outside the frame's outermost pixel centres.
    target = np.float32([[0, 0], [w, 0], [w, h], [0, h]]) - 0.5 - border
    warp = cv2.getPerspectiveTransform(corners.astype(np.float32), target)
    return cv2.warpPerspective(bgr, warp, (VIEW_W, VIEW_H), flags=cv2.INTER_LINEAR)


def locate(corners: np.ndarray, camera_matrix: np.ndarray, view_width_m: float = VIEW_WIDTH_M) -> Position | None:
    """Where the camera is, from the frame's corners in a pinhole picture.

    The frame is a rectangle of known size. Put a coordinate system on its
    centre: x to the right as you face the screen, y down, z into the screen.
    solvePnP finds the rotation R and translation t that take frame coordinates
    to camera coordinates; the camera then sits at -R^T t in the frame's
    coordinates.
    """
    half_w = view_width_m / 2
    half_h = half_w / FRAME_ASPECT
    frame_points = np.float32([[-half_w, -half_h, 0], [half_w, -half_h, 0], [half_w, half_h, 0], [-half_w, half_h, 0]])
    ok, rotation_vector, t = cv2.solvePnP(frame_points, corners.astype(np.float32), camera_matrix, None,
                                          flags=cv2.SOLVEPNP_IPPE)
    if not ok:
        return None
    rotation, _ = cv2.Rodrigues(rotation_vector)
    camera = (-rotation.T @ t).ravel()
    forward = rotation.T @ np.array([0.0, 0.0, 1.0])  # the camera's forward axis, in frame coordinates
    return Position(
        left=float(-camera[0]),       # the camera is at +x when it is to the RIGHT of the centre line
        distance=float(-camera[2]),   # in front of the screen is -z
        facing_right_deg=float(np.degrees(np.arctan2(forward[0], forward[2]))),
    )


@dataclass
class Sighting:
    view: np.ndarray           # the game view, for perception.detect
    position: Position | None  # where Pupper stands
    corners: np.ndarray        # the frame's corners in the pinhole picture
    picture: np.ndarray        # the pinhole picture (for the debug view)


class ScreenFinder:
    """Camera picture in, game view and position out. Remembers the fisheye maps."""

    def __init__(self, view_width_m: float = VIEW_WIDTH_M, rotate_180: bool = False):
        self.view_width_m = view_width_m
        self.rotate_180 = rotate_180
        self.pinhole: Pinhole | None = None
        self.size = None
        self.picture: np.ndarray | None = None  # the last pinhole picture, frame found or not

    def find(self, fisheye_bgr: np.ndarray) -> Sighting | None:
        if self.rotate_180:
            fisheye_bgr = cv2.rotate(fisheye_bgr, cv2.ROTATE_180)
        size = fisheye_bgr.shape[1], fisheye_bgr.shape[0]
        if self.pinhole is None or size != self.size:
            self.pinhole, self.size = Pinhole(*size), size
        picture = self.picture = self.pinhole(fisheye_bgr)
        corners = find_frame(picture)
        if corners is None:
            return None
        return Sighting(game_view(picture, corners), locate(corners, self.pinhole.matrix, self.view_width_m),
                        corners, picture)
