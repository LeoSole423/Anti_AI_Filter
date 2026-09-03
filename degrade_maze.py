#!/usr/bin/env python3
"""Procedural, deterministic visual degradation for maze raster images.

The structural masks are used as a guardrail: visual operations are never
allowed to remove the warped wall core, and distractors are kept out of the
protected corridor core.  This is a robustness/legibility tool, not a claim
that the resulting image is unrecoverable by machine vision.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from reportlab.lib.pagesizes import A4
from skimage.measure import euler_number
from skimage.morphology import skeletonize


LEVELS = ("subtle", "low", "medium", "strong", "max_readable")
LEVEL_PARAMS = {
    "subtle": dict(warp=0.12, texture=0.16, illumination=0.03, distractors=0.35, resample=0.90, jpeg=90, perspective=0.003, blur=0.20),
    "low": dict(warp=0.23, texture=0.25, illumination=0.04, distractors=0.55, resample=0.88, jpeg=88, perspective=0.005, blur=0.28),
    "medium": dict(warp=0.34, texture=0.36, illumination=0.05, distractors=0.80, resample=0.84, jpeg=85, perspective=0.008, blur=0.38),
    "strong": dict(warp=0.46, texture=0.48, illumination=0.065, distractors=1.05, resample=0.80, jpeg=82, perspective=0.011, blur=0.48),
    "max_readable": dict(warp=0.58, texture=0.60, illumination=0.08, distractors=1.30, resample=0.76, jpeg=80, perspective=0.014, blur=0.58),
}
MAX_VARIANT_ATTEMPTS = 10


@dataclass
class ImageAnalysis:
    width: int
    height: int
    channels: int
    roi: Tuple[int, int, int, int]
    otsu_threshold: float
    estimated_wall_width_px: float
    dark_fraction: float
    saturated_fraction: float
    marker_pixels: int


@dataclass
class ValidationResult:
    jacobian: bool
    wall_components: bool
    free_space_components: bool
    euler: bool
    no_crop: bool
    wall_core_integrity: bool
    jacobian_min: float
    original_wall_components: int
    candidate_wall_components: int
    original_free_components: int
    candidate_free_components: int
    original_euler: int
    candidate_euler: int
    wall_core_recall: float
    crop_recall: float

    @property
    def passed(self) -> bool:
        return all((self.jacobian, self.wall_components, self.free_space_components,
                    self.euler, self.no_crop, self.wall_core_integrity))


@dataclass(frozen=True)
class PerspectiveTransform:
    """One padded canvas and one homography shared by every image plane."""

    H: np.ndarray
    src: np.ndarray
    dst: np.ndarray
    pad: int
    strength: float
    output_shape: Tuple[int, int]
    crop_origin: Tuple[int, int]

    @property
    def crop_window(self) -> Tuple[int, int, int, int]:
        x, y = self.crop_origin
        h, w = self.output_shape
        return x, y, w, h


@dataclass
class AttemptResult:
    attempt: int
    scale: float
    amp_px: float
    perspective_used: float
    jacobian_min: float
    validation: ValidationResult
    failure_reasons: List[str]
    metadata: Dict


@dataclass
class VariantResult:
    level: str
    status: Literal["PASS", "FAIL"]
    image: Optional[np.ndarray]
    validation: Optional[ValidationResult]
    attempts: List[AttemptResult]
    failure_reasons: List[str]
    candidate_masks: Optional[Dict[str, np.ndarray]]


def _normalize_field(field: np.ndarray) -> np.ndarray:
    field = field.astype(np.float32)
    lo, hi = np.percentile(field, [1.0, 99.0])
    if hi <= lo:
        return np.zeros_like(field)
    return np.clip((field - lo) / (hi - lo) * 2.0 - 1.0, -1.0, 1.0)


def parse_background(value: str) -> Tuple[int, int, int]:
    """Parse an explicit RGB background in #rrggbb form for alpha flattening."""
    if not isinstance(value, str) or re.fullmatch(r"#[0-9a-fA-F]{6}", value) is None:
        raise argparse.ArgumentTypeError("--background debe tener formato hexadecimal #rrggbb, por ejemplo #ffffff")
    return tuple(int(value[i:i+2], 16) for i in (1, 3, 5))


def _background_hex(background: Tuple[int, int, int]) -> str:
    return "#" + "".join(f"{int(channel):02x}" for channel in background)


def load_image(path: str | Path, background: Tuple[int, int, int] = (255, 255, 255), return_metadata: bool = False):
    """Flatten input RGBA over an explicit RGB background before analysis."""
    source = Image.open(path)
    input_mode = source.mode
    had_alpha = "A" in source.getbands() or "transparency" in source.info
    src = source.convert("RGBA")
    bg = Image.new("RGBA", src.size, (*background, 255))
    flattened = Image.alpha_composite(bg, src).convert("RGB")
    rgb = np.asarray(flattened, dtype=np.uint8).copy()
    alpha = np.asarray(src.getchannel("A"), dtype=np.uint8).copy()
    if return_metadata:
        return rgb, alpha, {"input_mode": input_mode, "had_alpha": bool(had_alpha), "background": _background_hex(background)}
    return rgb, alpha


def parse_roi(roi: Optional[str], width: int, height: int) -> Optional[Tuple[int, int, int, int]]:
    if not roi:
        return None
    vals = [int(v.strip()) for v in roi.split(",")]
    if len(vals) != 4 or vals[2] <= 0 or vals[3] <= 0:
        raise ValueError("--roi debe tener el formato x,y,w,h con dimensiones positivas")
    x, y, w, h = vals
    if x < 0 or y < 0 or x + w > width or y + h > height:
        raise ValueError("--roi queda fuera de la imagen")
    return x, y, w, h


def detect_maze_roi(rgb: np.ndarray, manual_roi: Optional[Tuple[int, int, int, int]] = None) -> Tuple[int, int, int, int]:
    """Detect a dense rectangular maze area, with a reliable manual fallback."""
    if manual_roi is not None:
        return manual_roi
    h, w = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    otsu, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    dark = ((gray < max(100, min(220, otsu + 20))) & (hsv[..., 1] < 180)).astype(np.uint8)
    # Long dark runs identify the four frame sides in this kind of diagram.
    row_density = dark[:, int(0.02*w):int(0.98*w)].mean(axis=1)
    col_density = dark[int(0.06*h):int(0.94*h), :].mean(axis=0)
    row_candidates = np.flatnonzero(row_density > 0.32)
    col_candidates = np.flatnonzero(col_density > 0.32)
    top = int(row_candidates[0]) if len(row_candidates) else int(0.10*h)
    bottom = int(row_candidates[-1]) if len(row_candidates) else int(0.90*h)
    left = int(col_candidates[0]) if len(col_candidates) else int(0.02*w)
    right = int(col_candidates[-1]) if len(col_candidates) else int(0.98*w)
    # Keep a generous rectangle if projections were confused by title/footer.
    if bottom - top < 0.45*h or right - left < 0.45*w:
        top, bottom, left, right = int(0.10*h), int(0.90*h), int(0.02*w), int(0.98*w)
    # Inset only the outside text bands; preserve the frame itself.
    return max(0, left), max(0, top), min(w-left, right-left+1), min(h-top, bottom-top+1)


def detect_protected_markers(rgb: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    # Green/cyan marker range; saturation and value exclusions avoid dark ink.
    mask = ((hsv[..., 0] >= 32) & (hsv[..., 0] <= 95) & (hsv[..., 1] >= 85) & (hsv[..., 2] >= 90)).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def extract_wall_mask(rgb: np.ndarray, roi: Tuple[int, int, int, int], marker_mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, float]:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    otsu, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Ink is dark and low-saturation here.  The ROI removes title and footer.
    threshold = max(70, min(185, int(otsu + 18)))
    dark = (gray <= threshold) & (hsv[..., 1] < 180)
    x, y, w, h = roi
    roi_mask = np.zeros(gray.shape, np.uint8)
    roi_mask[y:y+h, x:x+w] = 1
    wall = (dark & (roi_mask > 0)).astype(np.uint8)
    if marker_mask is not None:
        wall[marker_mask > 0] = 0
    # Remove isolated text/particles while retaining long maze strokes.
    n, labels, stats, _ = cv2.connectedComponentsWithStats(wall, 8)
    keep = np.zeros_like(wall)
    min_area = max(20, int(0.00001 * gray.size))
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            keep[labels == i] = 1
    return keep, float(otsu)


def estimate_wall_width(wall_mask: np.ndarray) -> float:
    binary = wall_mask.astype(bool)
    if not binary.any():
        return 8.0
    skeleton = skeletonize(binary)
    distance = cv2.distanceTransform((binary * 255).astype(np.uint8), cv2.DIST_L2, 5)
    vals = distance[skeleton]
    vals = vals[(vals > 0.75) & (vals < np.percentile(vals, 95) if vals.size else True)]
    if vals.size == 0:
        return 8.0
    return float(max(2.0, 2.0 * np.median(vals)))


def build_safety_masks(wall_mask: np.ndarray, wall_width: float) -> Dict[str, np.ndarray]:
    wall = (wall_mask > 0).astype(np.uint8)
    radius = max(1, int(round(0.22 * wall_width)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radius+1, 2*radius+1))
    core = cv2.erode(wall, kernel, iterations=1)
    edge = cv2.subtract(wall, core)
    free = (wall == 0).astype(np.uint8)
    free_dist = cv2.distanceTransform(free * 255, cv2.DIST_L2, 5)
    corridor_core = (free_dist >= max(1.0, 0.65 * wall_width)).astype(np.uint8)
    return {"wall_mask": wall, "wall_core": core, "wall_edge": edge, "corridor_core": corridor_core}


def generate_displacement_field(shape: Tuple[int, int], rng: np.random.Generator, amplitude_px: float, sigma_fraction: float = 0.04) -> Tuple[np.ndarray, np.ndarray]:
    h, w = shape
    sigma = max(2.0, sigma_fraction * min(h, w))
    # OpenCV GaussianBlur requires a finite kernel; choose an odd size tied to sigma.
    k = max(3, int(round(6*sigma)) | 1)
    nx = cv2.GaussianBlur(rng.normal(size=(h, w)).astype(np.float32), (k, k), sigmaX=sigma)
    ny = cv2.GaussianBlur(rng.normal(size=(h, w)).astype(np.float32), (k, k), sigmaX=sigma)
    return _normalize_field(nx) * float(amplitude_px), _normalize_field(ny) * float(amplitude_px)


def validate_displacement_jacobian(dx: np.ndarray, dy: np.ndarray, minimum: float = 0.60) -> Tuple[bool, float, np.ndarray]:
    dxd_y, dxd_x = np.gradient(dx.astype(np.float32))
    dyd_y, dyd_x = np.gradient(dy.astype(np.float32))
    jac = (1.0 + dxd_x) * (1.0 + dyd_y) - dxd_y * dyd_x
    finite = jac[np.isfinite(jac)]
    min_j = float(finite.min()) if finite.size else 0.0
    return bool(min_j > minimum), min_j, jac


def apply_elastic_warp(image: np.ndarray, dx: np.ndarray, dy: np.ndarray, interpolation: int = cv2.INTER_LINEAR, border_mode: int = cv2.BORDER_REFLECT_101, border_value=0) -> np.ndarray:
    h, w = image.shape[:2]
    x, y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return cv2.remap(image, x + dx.astype(np.float32), y + dy.astype(np.float32), interpolation=interpolation, borderMode=border_mode, borderValue=border_value)


def build_perspective_transform(shape: Tuple[int, int], rng: np.random.Generator, strength: float, extra_margin_px: int = 0) -> PerspectiveTransform:
    """Generate exactly one deterministic padded homography for an attempt."""
    h, w = shape
    if strength <= 0:
        src = np.float32([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]])
        return PerspectiveTransform(np.eye(3, dtype=np.float32), src, src.copy(), 0, 0.0, (h, w), (0, 0))
    max_jitter = float(strength * min(h, w))
    pad = max(int(extra_margin_px), int(math.ceil(2.0 * max_jitter + 3.0)))
    src = np.float32([[pad, pad], [pad+w-1, pad], [pad+w-1, pad+h-1], [pad, pad+h-1]])
    jitter = rng.uniform(-strength, strength, size=(4, 2)).astype(np.float32) * min(h, w)
    dst = src + jitter
    H = cv2.getPerspectiveTransform(src, dst).astype(np.float32)
    # Choose the crop window around the transformed content, so a small global
    # translation does not clip a frame that originally touches the canvas.
    crop_x = int(round((float(dst[:, 0].min()) + float(dst[:, 0].max()) - (w-1)) / 2.0))
    crop_y = int(round((float(dst[:, 1].min()) + float(dst[:, 1].max()) - (h-1)) / 2.0))
    crop_x = max(0, min(crop_x, 2*pad)); crop_y = max(0, min(crop_y, 2*pad))
    return PerspectiveTransform(H, src, dst, pad, float(strength), (h, w), (crop_x, crop_y))


def apply_perspective_transform(arr: np.ndarray, transform: PerspectiveTransform, interpolation: int, border_value=0, crop: bool = True) -> np.ndarray:
    """Apply an already-built H; this function never samples a new jitter."""
    h, w = transform.output_shape
    pad = transform.pad
    if pad == 0:
        padded = arr
    else:
        if arr.ndim == 2:
            value = border_value if np.isscalar(border_value) else 0
        else:
            value = border_value if not np.isscalar(border_value) else (border_value,) * arr.shape[2]
        padded = cv2.copyMakeBorder(arr, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=value)
    out = cv2.warpPerspective(
        padded,
        transform.H,
        (w + 2*pad, h + 2*pad),
        flags=interpolation,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )
    if not crop:
        return out
    crop_x, crop_y = transform.crop_origin
    return out[crop_y:crop_y+h, crop_x:crop_x+w].copy()


def calculate_crop_recall(transformed_structural: np.ndarray, crop_window: Tuple[int, int, int, int]) -> float:
    """Return the transformed structural pixels retained by a crop window."""
    x, y, w, h = [int(v) for v in crop_window]
    structural = transformed_structural > 0
    expected = int(structural.sum())
    if expected == 0 or x < 0 or y < 0 or x+w > structural.shape[1] or y+h > structural.shape[0]:
        return 0.0
    visible = int(structural[y:y+h, x:x+w].sum())
    return float(visible / expected)


def _field(shape: Tuple[int, int], rng: np.random.Generator, sigma: float) -> np.ndarray:
    h, w = shape
    k = max(3, int(round(6*max(1.0, sigma))) | 1)
    return _normalize_field(cv2.GaussianBlur(rng.normal(size=(h, w)).astype(np.float32), (k, k), sigmaX=max(1.0, sigma)))


def apply_wall_texture(rgb: np.ndarray, masks: Dict[str, np.ndarray], rng: np.random.Generator, strength: float, protected: np.ndarray) -> np.ndarray:
    out = rgb.astype(np.float32).copy()
    edge = masks["wall_edge"] > 0
    if edge.any():
        h, w = edge.shape
        texture = 0.50*_field((h, w), rng, max(1.0, 0.0015*min(h,w))) + 0.30*_field((h,w), rng, max(2.0, 0.008*min(h,w))) + 0.20*_field((h,w), rng, max(3.0, 0.025*min(h,w)))
        factor = 1.0 + strength * 0.28 * texture
        out[edge] *= factor[edge, None]
        # Very soft pseudo-gaps only affect edge pixels; the core is reasserted later.
        scratches = (_field((h, w), rng, max(2.0, 0.006*min(h,w))) > 0.91) & edge
        out[scratches] = out[scratches] * (1.0 - 0.20*strength)
    out[protected > 0] = rgb[protected > 0]
    return np.clip(out, 0, 255).astype(np.uint8)


def generate_paper_texture(shape: Tuple[int, int], rng: np.random.Generator, strength: float) -> np.ndarray:
    h, w = shape
    fine = _field(shape, rng, max(0.7, 0.0008*min(h,w)))
    medium = _field(shape, rng, max(2.0, 0.008*min(h,w)))
    low = _field(shape, rng, max(4.0, 0.045*min(h,w)))
    texture = 0.50*fine + 0.30*medium + 0.20*low
    return texture.astype(np.float32) * float(3.0 + 10.0*strength)


def apply_illumination(rgb: np.ndarray, rng: np.random.Generator, strength: float, protected: np.ndarray) -> np.ndarray:
    h, w = rgb.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    linear = (xx/w - 0.5) * rng.uniform(-1, 1) + (yy/h - 0.5) * rng.uniform(-1, 1)
    radial = ((xx-w/2)**2 + (yy-h/2)**2) / ((w/2)**2 + (h/2)**2)
    low = _field((h,w), rng, max(4.0, 0.06*min(h,w)))
    delta = (linear + 0.45*radial + 0.55*low) * (255.0 * strength * 0.065)
    out = np.clip(rgb.astype(np.float32) + delta[..., None], 0, 255).astype(np.uint8)
    out[protected > 0] = rgb[protected > 0]
    return out


def add_safe_distractors(rgb: np.ndarray, masks: Dict[str, np.ndarray], wall_width: float, rng: np.random.Generator, amount: float, protected: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    out = rgb.copy()
    h, w = out.shape[:2]
    wall = masks["wall_mask"] > 0
    corridor = masks["corridor_core"] > 0
    core = masks["wall_core"] > 0
    dist_to_wall = cv2.distanceTransform((~wall).astype(np.uint8)*255, cv2.DIST_L2, 5)
    allowed = (~corridor) & (~core) & (~wall) & (dist_to_wall <= max(2.0, 2.0*wall_width))
    added = np.zeros((h,w), np.uint8)
    count = max(0, int(round(amount * (h*w) / (2048*2048) * 42)))
    max_len = max(2, int(round(1.5*wall_width)))
    min_len = max(2, int(round(0.3*wall_width)))
    for _ in range(count * 5):
        if int(added.sum()) >= count:
            break
        x = int(rng.integers(2, max(3, w-2)))
        y = int(rng.integers(2, max(3, h-2)))
        length = int(rng.integers(min_len, max_len+1))
        angle = float(rng.uniform(0, 2*np.pi))
        x2 = int(round(x + length*math.cos(angle)))
        y2 = int(round(y + length*math.sin(angle)))
        if not (0 <= x2 < w and 0 <= y2 < h):
            continue
        candidate = np.zeros((h,w), np.uint8)
        cv2.line(candidate, (x,y), (x2,y2), 1, max(1, int(round(rng.uniform(0.08,0.25)*wall_width)),), cv2.LINE_AA)
        pix = candidate > 0
        if not pix.any() or np.any(~allowed[pix]) or np.any(protected[pix] > 0):
            continue
        # Light grey fibres are intentionally far from ink darkness.
        tone = int(rng.integers(155, 205))
        out[pix] = (0.72*out[pix] + 0.28*tone).astype(np.uint8)
        added[pix] = 1
    return out, added


def apply_optical_degradation(rgb: np.ndarray, rng: np.random.Generator, sigma: float, protected: np.ndarray) -> np.ndarray:
    out = cv2.GaussianBlur(rgb, (0,0), sigmaX=max(0.01, sigma), sigmaY=max(0.01, sigma))
    # Tiny channel displacement, intentionally under one pixel.
    shift = min(1.0, max(0.0, sigma*1.25))
    if shift > 0.05:
        dx = float(rng.uniform(-shift, shift))
        dy = float(rng.uniform(-shift, shift))
        m = np.float32([[1,0,dx],[0,1,dy]])
        red = cv2.warpAffine(out[...,0], m, (out.shape[1],out.shape[0]), borderMode=cv2.BORDER_REFLECT_101)
        blue = cv2.warpAffine(out[...,2], m, (out.shape[1],out.shape[0]), borderMode=cv2.BORDER_REFLECT_101)
        out = out.copy(); out[...,0] = red; out[...,2] = blue
    return out if protected is None else out


def apply_resampling(rgb: np.ndarray, rng: np.random.Generator, ratio: float) -> np.ndarray:
    if ratio >= 0.999:
        return rgb.copy()
    h, w = rgb.shape[:2]
    nw, nh = max(32, int(round(w*ratio))), max(32, int(round(h*ratio)))
    small = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)


def _component_count(mask: np.ndarray) -> int:
    n, _, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
    areas = stats[1:, cv2.CC_STAT_AREA]
    return int(np.sum(areas > max(4, mask.size*1e-6))) if len(areas) else 0


def validate_topology(original_wall: np.ndarray, candidate_wall: np.ndarray, original_core: np.ndarray, candidate_core: np.ndarray, original_free: Optional[np.ndarray] = None, candidate_free: Optional[np.ndarray] = None, jacobian_min: float = 1.0, crop_recall: Optional[float] = None) -> ValidationResult:
    ow, cw = original_wall > 0, candidate_wall > 0
    of = ~ow if original_free is None else original_free > 0
    cf = ~cw if candidate_free is None else candidate_free > 0
    oc, cc = _component_count(ow), _component_count(cw)
    ofc, cfc = _component_count(of), _component_count(cf)
    oe = int(euler_number(ow, connectivity=2)); ce = int(euler_number(cw, connectivity=2))
    # The candidate core is already in the deformed coordinate system.  Compare
    # retained protected area, not same-coordinate overlap (a valid warp moves it).
    source_area = max(1, int((original_core > 0).sum()))
    candidate_area = int((candidate_core > 0).sum())
    recall = float(min(1.0, candidate_area / source_area))
    if crop_recall is None:
        crop_recall = float(cw.sum() / max(1, ow.sum()))
    no_crop = bool(crop_recall >= 0.995)
    return ValidationResult(
        jacobian=bool(jacobian_min > 0.60), wall_components=oc == cc,
        free_space_components=ofc == cfc, euler=oe == ce, no_crop=no_crop,
        wall_core_integrity=recall >= 0.985 and bool(np.all((candidate_core > 0) <= (candidate_wall > 0))), jacobian_min=float(jacobian_min),
        original_wall_components=oc, candidate_wall_components=cc,
        original_free_components=ofc, candidate_free_components=cfc,
        original_euler=oe, candidate_euler=ce, wall_core_recall=recall,
        crop_recall=float(crop_recall))


def run_recovery_proxies(rgb: np.ndarray, out_dir: Path) -> Dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    results = {}
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, a = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    results["A_otsu"] = str(out_dir / "A_otsu.png"); cv2.imwrite(results["A_otsu"], a)
    med = cv2.medianBlur(gray, 5)
    b = cv2.adaptiveThreshold(med,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,35,7)
    results["B_adaptive"] = str(out_dir / "B_adaptive.png"); cv2.imwrite(results["B_adaptive"], b)
    c = cv2.Canny(cv2.bilateralFilter(gray, 7, 35, 35), 50, 130)
    results["C_canny"] = str(out_dir / "C_canny.png"); cv2.imwrite(results["C_canny"], c)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(gray)
    _, d = cv2.threshold(clahe, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    d = cv2.morphologyEx(d, cv2.MORPH_CLOSE, np.ones((3,3),np.uint8))
    results["D_clahe_close"] = str(out_dir / "D_clahe_close.png"); cv2.imwrite(results["D_clahe_close"], d)
    lines = cv2.HoughLinesP(c, 1, np.pi/180, threshold=max(20, min(rgb.shape[:2])//20), minLineLength=max(10, min(rgb.shape[:2])//35), maxLineGap=max(3, min(rgb.shape[:2])//120))
    hough = np.zeros_like(gray)
    if lines is not None:
        for line in lines[:500]:
            x1,y1,x2,y2 = [int(v) for v in np.asarray(line).reshape(-1)[:4]]
            cv2.line(hough,(x1,y1),(x2,y2),255,1)
    results["E_hough"] = str(out_dir / "E_hough.png"); cv2.imwrite(results["E_hough"], hough)
    return results


def _save_rgb(path: Path, rgb: np.ndarray) -> None:
    Image.fromarray(rgb, mode="RGB").save(path)


def _label_font(size: int = 28):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def make_contact_sheet(images: Sequence[Tuple[str, np.ndarray]], path: Path) -> None:
    cell_w, cell_h = 650, 720
    sheet = Image.new("RGB", (cell_w*3, cell_h*2), (238,238,238))
    draw = ImageDraw.Draw(sheet); font = _label_font(30)
    for i,(label,rgb) in enumerate(images):
        x = (i%3)*cell_w; y = (i//3)*cell_h
        im = Image.fromarray(rgb).convert("RGB"); im.thumbnail((cell_w-24, cell_h-70), Image.Resampling.LANCZOS)
        px, py = x+(cell_w-im.width)//2, y+48+(cell_h-70-im.height)//2
        sheet.paste(im, (px,py)); draw.text((x+14,y+12), label.upper().replace("_"," "), fill=(20,20,20), font=font)
    sheet.save(path)


def make_debug_sheet(items: Sequence[Tuple[str, np.ndarray]], path: Path) -> None:
    cols = 4; cell_w, cell_h = 390, 330
    sheet = Image.new("RGB", (cols*cell_w, math.ceil(len(items)/cols)*cell_h), (235,235,235))
    draw = ImageDraw.Draw(sheet); font = _label_font(18)
    for i,(label,arr) in enumerate(items):
        x=(i%cols)*cell_w; y=(i//cols)*cell_h
        if arr.ndim == 2: arr=np.repeat(arr[...,None],3,axis=2)
        im=Image.fromarray(np.clip(arr,0,255).astype(np.uint8)).convert("RGB"); im.thumbnail((cell_w-10,cell_h-35),Image.Resampling.LANCZOS)
        sheet.paste(im,(x+(cell_w-im.width)//2,y+28)); draw.text((x+8,y+6),label,fill=(0,0,0),font=font)
    sheet.save(path)


def make_a4_raster(rgb: np.ndarray, path: Path, dpi: int = 300) -> None:
    W,H=2480,3508
    page=Image.new("RGB",(W,H),(255,255,255)); im=Image.fromarray(rgb).convert("RGB")
    margin=170; im.thumbnail((W-2*margin,H-2*margin),Image.Resampling.LANCZOS)
    page.paste(im,((W-im.width)//2,(H-im.height)//2)); page.save(path,dpi=(dpi,dpi))


def make_image_only_pdf(raster_path: Path, pdf_path: Path) -> None:
    c=canvas.Canvas(str(pdf_path),pagesize=A4)
    pw,ph=A4
    c.drawImage(ImageReader(str(raster_path)),0,0,width=pw,height=ph, preserveAspectRatio=False, mask="auto")
    c.showPage(); c.save()


def validation_failure_reasons(validation: ValidationResult) -> List[str]:
    """Return stable, machine-readable reasons for a failed attempt."""
    checks = (
        ("jacobian", validation.jacobian),
        ("wall_components", validation.wall_components),
        ("free_space_components", validation.free_space_components),
        ("euler", validation.euler),
        ("no_crop", validation.no_crop),
        ("wall_core_integrity", validation.wall_core_integrity),
    )
    return [name for name, passed in checks if not passed]


def _save_failed_variant_debug(debug_dir: Path, level: str, attempts: List[AttemptResult],
                               failure_reasons: List[str], payload: Optional[Tuple]) -> None:
    failed_dir = debug_dir / "failed" / level
    failed_dir.mkdir(parents=True, exist_ok=True)
    (failed_dir / "attempts.json").write_text(json.dumps({
        "level": level,
        "status": "FAIL",
        "failure_reasons": failure_reasons,
        "attempts": [asdict(item) for item in attempts],
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    (failed_dir / "failure_reasons.json").write_text(json.dumps(failure_reasons, indent=2, ensure_ascii=False), encoding="utf-8")
    if attempts:
        (failed_dir / "validation.json").write_text(json.dumps(asdict(attempts[-1].validation), indent=2, ensure_ascii=False), encoding="utf-8")
    if payload is None:
        return
    _, _, _, candidate_masks, dx, dy, jac, _ = payload
    cv2.imwrite(str(failed_dir / "final_displacement.png"), np.clip(
        np.dstack([_normalize_field(dx), _normalize_field(dy), np.zeros_like(dx)]) * 127.5 + 127.5,
        0, 255).astype(np.uint8))
    cv2.imwrite(str(failed_dir / "final_jacobian.png"), cv2.normalize(jac, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8))
    for name in ("wall_mask", "wall_core", "wall_edge", "corridor_core"):
        cv2.imwrite(str(failed_dir / f"final_{name}.png"), candidate_masks[name] * 255)


def generate_variant(original_rgb: np.ndarray, wall_mask: np.ndarray, masks: Dict[str,np.ndarray], marker: np.ndarray,
                    wall_width: float, level: str, seed: int, no_perspective: bool=False,
                    debug_dir: Optional[Path]=None, validator=validate_topology,
                    max_attempts: int=MAX_VARIANT_ATTEMPTS, save_success_debug: bool=True) -> VariantResult:
    p=LEVEL_PARAMS[level]; base_rng=np.random.default_rng(seed)
    amp=float(p["warp"]*wall_width)
    attempts: List[AttemptResult] = []
    last_payload = None
    for attempt in range(1, max_attempts + 1):
        rng=np.random.default_rng(seed + attempt*100003)
        scale=0.65**(attempt-1)
        dx,dy=generate_displacement_field(original_rgb.shape[:2],rng,amp*scale)
        jac_ok,jmin,jac=validate_displacement_jacobian(dx,dy)
        warped=apply_elastic_warp(original_rgb,dx,dy,cv2.INTER_LINEAR)
        wm=apply_elastic_warp(wall_mask,dx,dy,cv2.INTER_NEAREST,cv2.BORDER_CONSTANT,0)
        wc=apply_elastic_warp(masks["wall_core"],dx,dy,cv2.INTER_NEAREST,cv2.BORDER_CONSTANT,0)
        we=apply_elastic_warp(masks["wall_edge"],dx,dy,cv2.INTER_NEAREST,cv2.BORDER_CONSTANT,0)
        cor=apply_elastic_warp(masks["corridor_core"],dx,dy,cv2.INTER_NEAREST,cv2.BORDER_CONSTANT,0)
        mm=apply_elastic_warp(marker,dx,dy,cv2.INTER_NEAREST,cv2.BORDER_CONSTANT,0)
        perspective_requested=not no_perspective
        perspective_used=p["perspective"]*scale if perspective_requested else 0.0
        perspective_transform=build_perspective_transform(original_rgb.shape[:2], rng, perspective_used,
                                                           extra_margin_px=max(2, int(round(wall_width))))
        warped=apply_perspective_transform(warped,perspective_transform,cv2.INTER_LINEAR,border_value=(255,255,255))
        wm_full=apply_perspective_transform(wm,perspective_transform,cv2.INTER_NEAREST,border_value=0,crop=False)
        wm=apply_perspective_transform(wm,perspective_transform,cv2.INTER_NEAREST,border_value=0)
        wc=apply_perspective_transform(wc,perspective_transform,cv2.INTER_NEAREST,border_value=0)
        we=apply_perspective_transform(we,perspective_transform,cv2.INTER_NEAREST,border_value=0)
        cor=apply_perspective_transform(cor,perspective_transform,cv2.INTER_NEAREST,border_value=0)
        mm=apply_perspective_transform(mm,perspective_transform,cv2.INTER_NEAREST,border_value=0)
        core_bin=(wc>0).astype(np.uint8)
        structural=np.maximum((wm>0).astype(np.uint8), core_bin)
        edge_bin=np.maximum((we>0).astype(np.uint8),cv2.subtract(structural,core_bin))
        corridor_bin=cv2.subtract((cor>0).astype(np.uint8),structural)
        candidate_masks={"wall_mask":structural,"wall_core":core_bin,"wall_edge":edge_bin,"corridor_core":corridor_bin}
        crop_recall=calculate_crop_recall(wm_full,perspective_transform.crop_window)
        val=validator(wall_mask,candidate_masks["wall_mask"],masks["wall_core"],candidate_masks["wall_core"],
                      jacobian_min=jmin,crop_recall=crop_recall)
        val.jacobian = jac_ok
        metadata=dict(attempt=attempt, scale=scale, amp_px=amp*scale, jacobian_min=jmin,
            perspective_requested=perspective_requested, perspective_used=perspective_used,
            src_corners=perspective_transform.src.tolist(), dst_corners=perspective_transform.dst.tolist(),
            H=perspective_transform.H.tolist(), pad=perspective_transform.pad,
            crop_origin=list(perspective_transform.crop_origin), crop_x=perspective_transform.crop_window[0],
            crop_y=perspective_transform.crop_window[1], crop_width=perspective_transform.crop_window[2],
            crop_height=perspective_transform.crop_window[3], crop_recall=crop_recall)
        reasons=validation_failure_reasons(val)
        attempt_result=AttemptResult(attempt, scale, amp*scale, perspective_used, jmin, val, reasons, metadata)
        attempts.append(attempt_result)
        last_payload=(warped,val,metadata,candidate_masks,dx,dy,jac,mm)
        if val.passed and not reasons:
            # Only structurally validated candidates reach visual degradation.
            out=warped.copy()
            out=np.clip(out.astype(np.float32)+generate_paper_texture(out.shape[:2],base_rng,p["texture"])[...,None],0,255).astype(np.uint8)
            out=apply_wall_texture(out,candidate_masks,base_rng,p["texture"],mm)
            out=apply_illumination(out,base_rng,p["illumination"],mm)
            out,distractors=add_safe_distractors(out,candidate_masks,wall_width,base_rng,p["distractors"],mm)
            out=apply_optical_degradation(out,base_rng,p["blur"],mm)
            out=apply_resampling(out,base_rng,p["resample"])
            ink_color=np.median(warped[candidate_masks["wall_core"]>0],axis=0) if np.any(candidate_masks["wall_core"]>0) else np.array([30,38,55])
            out[candidate_masks["wall_core"]>0]=np.minimum(out[candidate_masks["wall_core"]>0], np.clip(ink_color,15,90).astype(np.uint8))
            out[mm>0]=warped[mm>0]
            if debug_dir is not None and save_success_debug:
                debug_dir.mkdir(parents=True,exist_ok=True)
                cv2.imwrite(str(debug_dir/f"displacement_{level}.png"),np.clip(np.dstack([_normalize_field(dx),_normalize_field(dy),np.zeros_like(dx)])*127.5+127.5,0,255).astype(np.uint8))
                cv2.imwrite(str(debug_dir/f"jacobian_{level}.png"),cv2.normalize(jac,None,0,255,cv2.NORM_MINMAX).astype(np.uint8))
                cv2.imwrite(str(debug_dir/f"wall_mask_{level}.png"),candidate_masks["wall_mask"]*255)
                cv2.imwrite(str(debug_dir/f"wall_core_{level}.png"),candidate_masks["wall_core"]*255)
                cv2.imwrite(str(debug_dir/f"corridor_core_{level}.png"),candidate_masks["corridor_core"]*255)
                cv2.imwrite(str(debug_dir/f"distractors_{level}.png"),distractors*255)
            metadata["wall_core_recall"]=val.wall_core_recall
            return VariantResult(level, "PASS", out, val, attempts, [], candidate_masks)
    reasons=[]
    for item in attempts:
        for reason in item.failure_reasons:
            if reason not in reasons:
                reasons.append(reason)
    result=VariantResult(level, "FAIL", None, attempts[-1].validation if attempts else None, attempts, reasons, None)
    if debug_dir is not None:
        _save_failed_variant_debug(debug_dir, level, attempts, reasons, last_payload)
    return result


def _parse_levels(raw: str) -> List[str]:
    if raw.lower() in ("all", "*"):
        return list(LEVELS)
    values=[x.strip().lower() for x in raw.split(",") if x.strip()]
    unknown=set(values)-set(LEVELS)
    if unknown: raise ValueError(f"Niveles desconocidos: {', '.join(sorted(unknown))}")
    return [x for x in LEVELS if x in values]


def main(argv: Optional[Sequence[str]]=None) -> int:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input", type=Path); ap.add_argument("--output-dir", type=Path, default=Path("output")); ap.add_argument("--seed", type=int, default=423)
    ap.add_argument("--roi", help="x,y,w,h; si falta se detecta automaticamente"); ap.add_argument("--levels", default="all", help="all o lista separada por comas")
    ap.add_argument("--pdf", action="store_true", help="generar también el PDF A4 raster"); ap.add_argument("--debug", action="store_true", help="guardar máscaras, campos y recuperaciones"); ap.add_argument("--no-perspective", action="store_true")
    ap.add_argument("--background", type=parse_background, default=(255,255,255), help="fondo RGB para aplanar transparencia, formato #rrggbb")
    args=ap.parse_args(argv)
    if not args.input.exists(): ap.error(f"No existe la imagen: {args.input}")
    if args.output_dir.resolve() == args.input.resolve().parent: ap.error("--output-dir no puede ser la carpeta de la imagen original")
    levels=_parse_levels(args.levels); args.output_dir.mkdir(parents=True,exist_ok=True)
    debug=args.output_dir/"debug"; debug.mkdir(exist_ok=True)
    rgb,alpha,load_info=load_image(args.input,args.background,return_metadata=True); h,w=rgb.shape[:2]
    roi=parse_roi(args.roi,w,h); roi=detect_maze_roi(rgb,roi)
    marker=detect_protected_markers(rgb); wall,otsu=extract_wall_mask(rgb,roi,marker); wall_width=estimate_wall_width(wall); masks=build_safety_masks(wall,wall_width)
    analysis=ImageAnalysis(w,h,rgb.shape[2],roi,otsu,wall_width,float((wall>0).mean()),float((cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)[...,1]>85).mean()),int(marker.sum()))
    _save_rgb(debug/"wall_mask_original.png",np.repeat((wall*255)[...,None],3,axis=2)); _save_rgb(debug/"wall_core_original.png",np.repeat((masks["wall_core"]*255)[...,None],3,axis=2)); _save_rgb(debug/"corridor_core_original.png",np.repeat((masks["corridor_core"]*255)[...,None],3,axis=2))
    variants=[]; validations={}; attempts={}; results={}; failed_levels=[]
    original_for_sheet=rgb.copy()
    for idx,level in enumerate(levels,1):
        result=generate_variant(rgb,wall,masks,marker,wall_width,level,args.seed+idx*7919,args.no_perspective,debug,save_success_debug=args.debug)
        results[level]=result
        validations[level]=asdict(result.validation) if result.validation is not None else None
        attempts[level]=result.attempts[-1].metadata if result.attempts else {}
        if result.status == "PASS" and result.image is not None and result.validation is not None and result.validation.passed:
            out=result.image
            png=args.output_dir/f"maze_{idx:02d}_{level}.png"; jpg=args.output_dir/f"maze_{idx:02d}_{level}.jpg"
            _save_rgb(png,out); Image.fromarray(out).save(jpg,quality=LEVEL_PARAMS[level]["jpeg"],optimize=True,subsampling=0)
            if args.debug: run_recovery_proxies(out,debug/"recovery"/level)
            variants.append((level,out))
        else:
            # Remove only this run's publish targets, preventing stale files
            # from making a failed variant look published after a rerun.
            (args.output_dir/f"maze_{idx:02d}_{level}.png").unlink(missing_ok=True)
            (args.output_dir/f"maze_{idx:02d}_{level}.jpg").unlink(missing_ok=True)
            failed_levels.append(level)
    if len(levels)==len(LEVELS):
        contact_variants=[]
        for level in levels:
            result=results[level]
            if result.status == "PASS" and result.image is not None:
                contact_variants.append((level,result.image))
            else:
                contact_variants.append((f"{level} FAILED",np.full_like(original_for_sheet,235)))
        make_contact_sheet([("original",original_for_sheet)]+contact_variants,args.output_dir/"contact_sheet.png")
    debug_items=[("original",rgb),("wall mask",wall*255),("wall core",masks["wall_core"]*255),("corridor core",masks["corridor_core"]*255)]
    if args.debug:
        for level,out in variants:
            debug_items.append((level,out))
            pth=debug/"recovery"/level/"A_otsu.png"
            if pth.exists(): debug_items.append((f"{level} A otsu",np.asarray(Image.open(pth).convert("L"))))
            for prefix in ("displacement_", "jacobian_"):
                pth=debug/f"{prefix}{level}.png"
                if pth.exists(): debug_items.append((f"{level} {prefix[:-1]}",np.asarray(Image.open(pth).convert("L"))))
    make_debug_sheet(debug_items, args.output_dir/"debug_sheet.png")
    valid_levels=[lv for lv in levels if lv in results and results[lv].status == "PASS" and results[lv].image is not None and results[lv].validation is not None and results[lv].validation.passed]
    recommended=("medium" if "medium" in valid_levels else ("strong" if "strong" in valid_levels else (valid_levels[0] if valid_levels else None)))
    pdf_generated=False
    if recommended is not None:
        rec_rgb=dict(variants)[recommended]
        raster=args.output_dir/"maze_recommended_A4.png"; make_a4_raster(rec_rgb,raster)
        if args.pdf:
            make_image_only_pdf(raster,args.output_dir/"maze_recommended_A4.pdf")
            pdf_generated=True
    else:
        # A failed run must not leave a stale recommendation from an earlier run.
        (args.output_dir/"maze_recommended_A4.png").unlink(missing_ok=True)
        (args.output_dir/"maze_recommended_A4.pdf").unlink(missing_ok=True)
    variant_records={level:{"status":result.status,"failure_reasons":result.failure_reasons,
                            "validation":asdict(result.validation) if result.validation is not None else None,
                            "attempts":[asdict(item) for item in result.attempts]}
                     for level,result in results.items()}
    run_status="FAIL" if not valid_levels else ("PARTIAL_SUCCESS" if failed_levels else "PASS")
    config={"input":str(args.input.resolve()),"output_dir":str(args.output_dir.resolve()),"seed":args.seed,"roi":roi,"analysis":asdict(analysis),"input_mode":load_info["input_mode"],"had_alpha":load_info["had_alpha"],"background":load_info["background"],"levels":levels,"parameters":LEVEL_PARAMS,"no_perspective":args.no_perspective,"perspective_requested":not args.no_perspective,"status":run_status,"partial_success":bool(valid_levels and failed_levels),"valid_levels":valid_levels,"failed_levels":failed_levels,"validations":validations,"attempts":attempts,"variants":variant_records,"recommended":recommended,"pdf_generated":pdf_generated}
    (args.output_dir/"run_config.json").write_text(json.dumps(config,indent=2,ensure_ascii=False),encoding="utf-8")
    print(json.dumps({"status":run_status,"roi":roi,"estimated_wall_width_px":wall_width,"recommended":recommended,"valid_levels":valid_levels,"failed_levels":failed_levels,"validations":validations},ensure_ascii=False,indent=2))
    return 0 if valid_levels else 1


if __name__=="__main__":
    raise SystemExit(main())
