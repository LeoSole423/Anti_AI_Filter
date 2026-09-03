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
class ComponentCorrespondenceResult:
    passed: bool
    splits: int
    merges: int
    missing: int
    unexpected: int
    min_expected_coverage: float
    min_candidate_purity: float
    per_label: Dict


@dataclass
class SeedValidationResult:
    seed_count: int
    misses: int
    collisions: int
    per_label: Dict


@dataclass
class SkeletonSignature:
    endpoints: int
    junctions: int


@dataclass
class PostRenderValidation:
    passed: bool
    contrast_gap: float
    threshold: float
    min_seed_clearance_px: float
    p05_corridor_clearance_px: float
    seed_misses: int
    seed_collisions: int
    wall_identity: ComponentCorrespondenceResult
    free_identity: ComponentCorrespondenceResult


@dataclass
class TopologyReference:
    domain_mask: np.ndarray
    wall_labels: np.ndarray
    free_labels: np.ndarray
    wall_ids: List[int]
    free_ids: List[int]
    wall_core: np.ndarray
    min_component_area: int


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
    wall_core_precision: float = 1.0
    wall_label_identity: bool = True
    free_label_identity: bool = True
    core_geometry: bool = True
    post_render_topology: bool = True
    wall_identity: Optional[ComponentCorrespondenceResult] = None
    free_identity: Optional[ComponentCorrespondenceResult] = None
    wall_seeds: Optional[SeedValidationResult] = None
    free_seeds: Optional[SeedValidationResult] = None
    skeleton_original: Optional[SkeletonSignature] = None
    skeleton_candidate: Optional[SkeletonSignature] = None
    post_render: Optional[PostRenderValidation] = None

    def gate_checks(self) -> Dict[str, bool]:
        return {
            "jacobian": self.jacobian,
            "wall_components": self.wall_components,
            "free_space_components": self.free_space_components,
            "euler": self.euler,
            "no_crop": self.no_crop,
            "wall_core_integrity": self.wall_core_integrity,
            "wall_label_identity": self.wall_label_identity,
            "free_label_identity": self.free_label_identity,
            "core_geometry": self.core_geometry,
            "post_render_topology": self.post_render_topology,
        }

    @property
    def passed(self) -> bool:
        return all(self.gate_checks().values())


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


def _filtered_component_labels(mask: np.ndarray, min_area: Optional[int] = None) -> Tuple[np.ndarray, List[int], int]:
    """Label only components accepted by the existing component-size policy."""
    binary = (mask > 0).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    minimum = int(min_area if min_area is not None else max(4, binary.size * 1e-6))
    ids = [int(i) for i in range(1, n) if int(stats[i, cv2.CC_STAT_AREA]) >= minimum]
    if not ids:
        return np.zeros_like(binary, dtype=np.uint16), [], minimum
    accepted = np.isin(labels, ids)
    return np.where(accepted, labels, 0).astype(np.uint16), ids, minimum


def build_topology_reference(wall_mask: np.ndarray, wall_core: np.ndarray,
                             domain_mask: Optional[np.ndarray] = None) -> TopologyReference:
    """Create the one identity-preserving wall/free reference used by a run."""
    domain = np.ones_like(wall_mask, dtype=np.uint8) if domain_mask is None else (domain_mask > 0).astype(np.uint8)
    wall = ((wall_mask > 0) & (domain > 0)).astype(np.uint8)
    free = ((domain > 0) & ~(wall > 0)).astype(np.uint8)
    wall_labels, wall_ids, minimum = _filtered_component_labels(wall)
    free_labels, free_ids, _ = _filtered_component_labels(free, minimum)
    return TopologyReference(domain, wall_labels, free_labels, wall_ids, free_ids,
                             ((wall_core > 0) & (domain > 0)).astype(np.uint8), minimum)


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


def warp_label_map(labels: np.ndarray, dx: np.ndarray, dy: np.ndarray,
                   perspective_transform: PerspectiveTransform) -> np.ndarray:
    """Warp integer identities with nearest-neighbour and constant background."""
    labels = np.asarray(labels, dtype=np.uint16)
    elastic = apply_elastic_warp(labels, dx, dy, cv2.INTER_NEAREST, cv2.BORDER_CONSTANT, 0)
    warped = apply_perspective_transform(elastic, perspective_transform, cv2.INTER_NEAREST,
                                         border_value=0)
    return np.rint(warped).astype(np.uint16)


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
    _, ids, _ = _filtered_component_labels(mask)
    return len(ids)


def compare_component_identity(expected_labels: np.ndarray, candidate_binary: np.ndarray,
                               valid_domain: np.ndarray, min_component_area: int,
                               require_quality: bool=True) -> ComponentCorrespondenceResult:
    """Compare expected identities to candidate components through overlap."""
    expected = np.asarray(expected_labels, dtype=np.uint16)
    domain = (valid_domain > 0)
    candidate_labels, candidate_ids, _ = _filtered_component_labels(
        (candidate_binary > 0) & domain, min_component_area)
    expected_ids = [int(x) for x in np.unique(expected[domain]) if x > 0]
    expected_areas = {eid: int(np.count_nonzero((expected == eid) & domain)) for eid in expected_ids}
    candidate_areas = {cid: int(np.count_nonzero(candidate_labels == cid)) for cid in candidate_ids}
    expected_to_candidates: Dict[int, List[Tuple[int, int]]] = {}
    candidate_to_expected: Dict[int, List[Tuple[int, int]]] = {}
    for eid in expected_ids:
        values, counts = np.unique(candidate_labels[(expected == eid) & domain], return_counts=True)
        significant = max(3, int(round(0.005 * expected_areas[eid])))
        expected_to_candidates[eid] = [(int(cid), int(count)) for cid, count in zip(values, counts)
                                       if cid > 0 and int(count) >= significant]
    for cid in candidate_ids:
        values, counts = np.unique(expected[(candidate_labels == cid) & domain], return_counts=True)
        significant = max(3, int(round(0.005 * candidate_areas[cid])))
        candidate_to_expected[cid] = [(int(eid), int(count)) for eid, count in zip(values, counts)
                                      if eid > 0 and int(count) >= significant]

    splits = sum(len(matches) > 1 for matches in expected_to_candidates.values())
    missing = sum(len(matches) == 0 for matches in expected_to_candidates.values())
    merges = sum(len(matches) > 1 for matches in candidate_to_expected.values())
    unexpected = sum(len(matches) == 0 for matches in candidate_to_expected.values())
    coverages = []
    for eid, matches in expected_to_candidates.items():
        dominant = max((count for _, count in matches), default=0)
        coverages.append(dominant / max(1, expected_areas[eid]))
    purities = []
    for cid, matches in candidate_to_expected.items():
        dominant = max((count for _, count in matches), default=0)
        purities.append(dominant / max(1, candidate_areas[cid]))
    per_label = {
        "expected": {str(eid): {"area": expected_areas[eid], "matches": matches}
                     for eid, matches in expected_to_candidates.items()},
        "candidate": {str(cid): {"area": candidate_areas[cid], "matches": matches}
                       for cid, matches in candidate_to_expected.items()},
    }
    min_coverage = float(min(coverages)) if coverages else 1.0
    min_purity = float(min(purities)) if purities else 1.0
    passed = (splits == 0 and merges == 0 and missing == 0 and unexpected == 0
              and (not require_quality or (min_coverage >= 0.98 and min_purity >= 0.98)))
    return ComponentCorrespondenceResult(bool(passed), splits, merges, missing, unexpected,
                                         min_coverage, min_purity, per_label)


def _interior_seeds(expected_labels: np.ndarray, ids: Sequence[int], valid_domain: np.ndarray) -> Dict[int, List[Tuple[int, int]]]:
    seeds: Dict[int, List[Tuple[int, int]]] = {}
    domain = valid_domain > 0
    for label_id in ids:
        component = ((expected_labels == int(label_id)) & domain).astype(np.uint8)
        area = int(component.sum())
        if area == 0:
            seeds[int(label_id)] = []
            continue
        distance = cv2.distanceTransform(component * 255, cv2.DIST_L2, 5)
        count = 3 if area >= 1000 else 1
        chosen: List[Tuple[int, int]] = []
        min_separation = max(3.0, math.sqrt(area) * 0.10)
        working = distance.copy()
        for _ in range(count):
            _, maximum, _, location = cv2.minMaxLoc(working)
            if maximum <= 0:
                break
            x, y = int(location[0]), int(location[1])
            chosen.append((x, y))
            cv2.circle(working, (x, y), int(math.ceil(min_separation)), 0, -1)
        seeds[int(label_id)] = chosen
    return seeds


def validate_seed_correspondence(expected_labels: np.ndarray, ids: Sequence[int],
                                 candidate_binary: np.ndarray, valid_domain: np.ndarray,
                                 min_component_area: int) -> SeedValidationResult:
    candidate_labels, _, _ = _filtered_component_labels(
        (candidate_binary > 0) & (valid_domain > 0), min_component_area)
    seeds = _interior_seeds(expected_labels, ids, valid_domain)
    misses = 0
    candidates_by_expected: Dict[int, List[int]] = {}
    for expected_id, points in seeds.items():
        values = [int(candidate_labels[y, x]) for x, y in points]
        nonzero = sorted(set(value for value in values if value > 0))
        if not points or len(nonzero) != 1 or any(value == 0 for value in values):
            misses += 1
        candidates_by_expected[expected_id] = nonzero
    collisions = 0
    reverse: Dict[int, List[int]] = {}
    for expected_id, candidate_ids in candidates_by_expected.items():
        for candidate_id in candidate_ids:
            reverse.setdefault(candidate_id, []).append(expected_id)
    collisions = sum(len(expected_ids) > 1 for expected_ids in reverse.values())
    return SeedValidationResult(sum(len(points) for points in seeds.values()), misses,
                                collisions, {str(k): v for k, v in candidates_by_expected.items()})


def skeleton_signature(mask: np.ndarray) -> SkeletonSignature:
    skeleton = skeletonize(mask > 0)
    neighbours = cv2.filter2D(skeleton.astype(np.uint8), cv2.CV_16U,
                              np.ones((3, 3), np.uint8)) - skeleton.astype(np.uint8)
    endpoints = int(np.count_nonzero(skeleton & (neighbours == 1)))
    junction_pixels = (skeleton & (neighbours >= 3)).astype(np.uint8)
    junction_components, _ = cv2.connectedComponents(junction_pixels, 8)
    return SkeletonSignature(endpoints, max(0, int(junction_components - 1)))


def _identity_debug(result: Optional[ComponentCorrespondenceResult]) -> Dict:
    return asdict(result) if result is not None else {}


def validate_post_render(rgb: np.ndarray, topology: TopologyReference,
                         expected_wall_labels: np.ndarray, expected_free_labels: np.ndarray,
                         expected_domain: np.ndarray, candidate_masks: Dict[str, np.ndarray],
                         marker: np.ndarray, distractor_mask: Optional[np.ndarray]=None) -> PostRenderValidation:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    protected_marker = marker > 0
    wall_pixels = (candidate_masks["wall_core"] > 0) & ~protected_marker
    free_pixels = (candidate_masks["corridor_core"] > 0) & ~protected_marker
    wall_hi = float(np.percentile(gray[wall_pixels], 90)) if wall_pixels.any() else 100.0
    free_lo = float(np.percentile(gray[free_pixels], 10)) if free_pixels.any() else 220.0
    contrast_gap = free_lo - wall_hi
    # Bias the supervised segmentation toward retaining ink.  A midpoint is
    # vulnerable to anti-aliased wall edges becoming free-space leaks after
    # resampling; this conservative threshold remains below the measured safe
    # corridor luminance and does not manufacture dark pixels.
    domain = (expected_domain > 0).astype(np.uint8)
    thresholds = [wall_hi + fraction * (free_lo - wall_hi) for fraction in (0.50, 0.70, 0.85)]
    evaluations: List[PostRenderValidation] = []
    for threshold in thresholds:
        render_wall = (gray <= threshold).astype(np.uint8)
        render_wall[protected_marker] = 0
        # Known light fibres are visual artifacts, not structural ink.  Excluding
        # their exact procedural mask prevents the conservative threshold from
        # turning them into unexpected maze components.
        if distractor_mask is not None:
            render_wall[distractor_mask > 0] = 0
        render_free = cv2.bitwise_and(domain, (render_wall == 0).astype(np.uint8))
        wall_identity = compare_component_identity(expected_wall_labels, render_wall, domain,
                                                   topology.min_component_area, require_quality=False)
        free_identity = compare_component_identity(expected_free_labels, render_free, domain,
                                                   topology.min_component_area, require_quality=False)
        free_seeds = validate_seed_correspondence(expected_free_labels, topology.free_ids,
                                                   render_free, domain, topology.min_component_area)
        clearance = cv2.distanceTransform(render_free * 255, cv2.DIST_L2, 5)
        seed_points = _interior_seeds(expected_free_labels, topology.free_ids, domain)
        seed_clearances = [float(clearance[y, x]) for points in seed_points.values() for x, y in points]
        min_clearance = float(min(seed_clearances)) if seed_clearances else 0.0
        free_distances = clearance[render_free > 0]
        p05_clearance = float(np.percentile(free_distances, 5)) if free_distances.size else 0.0
        passed = bool(contrast_gap >= 20.0 and wall_identity.passed and free_identity.passed
                      and free_seeds.misses == 0 and free_seeds.collisions == 0 and min_clearance > 0.0)
        evaluations.append(PostRenderValidation(passed, contrast_gap, threshold, min_clearance,
                                                 p05_clearance, free_seeds.misses,
                                                 free_seeds.collisions, wall_identity, free_identity))
        if passed:
            return evaluations[-1]
    # Preserve the most informative failed segmentation in diagnostics.
    def score(item: PostRenderValidation) -> Tuple[int, int, float]:
        failures = (not item.wall_identity.passed) + (not item.free_identity.passed) + bool(item.seed_misses) + bool(item.seed_collisions) + (item.min_seed_clearance_px <= 0)
        unexpected = item.wall_identity.unexpected + item.free_identity.unexpected
        return int(failures), int(unexpected), -item.min_seed_clearance_px
    return min(evaluations, key=score)


def _label_debug_image(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.uint16)
    if int(labels.max()) == 0:
        return np.zeros((*labels.shape, 3), dtype=np.uint8)
    scaled = cv2.normalize(labels, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.applyColorMap(scaled, cv2.COLORMAP_TURBO)


def save_topology_debug(debug_dir: Path, level: str, expected_wall_labels: np.ndarray,
                        candidate_wall: np.ndarray, expected_free_labels: np.ndarray,
                        candidate_free: np.ndarray, render_wall: Optional[np.ndarray],
                        wall_identity: Optional[ComponentCorrespondenceResult],
                        free_identity: Optional[ComponentCorrespondenceResult],
                        post_render: Optional[PostRenderValidation]) -> None:
    topology_dir = debug_dir / "topology" / level
    topology_dir.mkdir(parents=True, exist_ok=True)
    candidate_wall_labels, _, _ = _filtered_component_labels(candidate_wall)
    candidate_free_labels, _, _ = _filtered_component_labels(candidate_free)
    cv2.imwrite(str(topology_dir / "expected_wall_labels.png"), _label_debug_image(expected_wall_labels))
    cv2.imwrite(str(topology_dir / "candidate_wall_labels.png"), _label_debug_image(candidate_wall_labels))
    cv2.imwrite(str(topology_dir / "expected_free_labels.png"), _label_debug_image(expected_free_labels))
    cv2.imwrite(str(topology_dir / "candidate_free_labels.png"), _label_debug_image(candidate_free_labels))
    if render_wall is not None:
        cv2.imwrite(str(topology_dir / "render_wall_mask.png"), (render_wall > 0).astype(np.uint8) * 255)
    (topology_dir / "correspondence_wall.json").write_text(json.dumps(
        _identity_debug(wall_identity), indent=2, ensure_ascii=False), encoding="utf-8")
    (topology_dir / "correspondence_free.json").write_text(json.dumps(
        _identity_debug(free_identity), indent=2, ensure_ascii=False), encoding="utf-8")
    if post_render is not None:
        (topology_dir / "post_render.json").write_text(json.dumps(
            asdict(post_render), indent=2, ensure_ascii=False), encoding="utf-8")


def validate_topology(original_wall: np.ndarray, candidate_wall: np.ndarray, original_core: np.ndarray,
                      candidate_core: np.ndarray, original_free: Optional[np.ndarray] = None,
                      candidate_free: Optional[np.ndarray] = None, jacobian_min: float = 1.0,
                      crop_recall: Optional[float] = None, topology: Optional[TopologyReference] = None,
                      expected_wall_labels: Optional[np.ndarray] = None,
                      expected_free_labels: Optional[np.ndarray] = None,
                      expected_domain: Optional[np.ndarray] = None,
                      expected_core: Optional[np.ndarray] = None,
                      post_render: Optional[PostRenderValidation] = None) -> ValidationResult:
    ow, cw = original_wall > 0, candidate_wall > 0
    of = ~ow if original_free is None else original_free > 0
    cf = ~cw if candidate_free is None else candidate_free > 0
    oc, cc = _component_count(ow), _component_count(cw)
    ofc, cfc = _component_count(of), _component_count(cf)
    oe = int(euler_number(ow, connectivity=2)); ce = int(euler_number(cw, connectivity=2))
    geometric_core = expected_core is not None
    expected_core = original_core if expected_core is None else expected_core
    source_area = max(1, int((expected_core > 0).sum()))
    candidate_area = int((candidate_core > 0).sum())
    intersection = int(np.count_nonzero((expected_core > 0) & (candidate_core > 0)))
    recall = float(intersection / source_area) if geometric_core else float(min(1.0, candidate_area / source_area))
    precision = float(intersection / max(1, candidate_area)) if geometric_core else float(min(1.0, source_area / max(1, candidate_area)))
    if crop_recall is None:
        crop_recall = float(cw.sum() / max(1, ow.sum()))
    no_crop = bool(crop_recall >= 0.995)
    wall_identity = None
    free_identity = None
    wall_seeds = None
    free_seeds = None
    if topology is not None and expected_wall_labels is not None and expected_free_labels is not None and expected_domain is not None:
        candidate_domain = expected_domain > 0
        wall_identity = compare_component_identity(expected_wall_labels, candidate_wall, candidate_domain, topology.min_component_area)
        free_identity = compare_component_identity(expected_free_labels, candidate_free if candidate_free is not None else ~candidate_wall, candidate_domain, topology.min_component_area)
        wall_seeds = validate_seed_correspondence(expected_wall_labels, topology.wall_ids, candidate_wall, candidate_domain, topology.min_component_area)
        free_seeds = validate_seed_correspondence(expected_free_labels, topology.free_ids, candidate_free if candidate_free is not None else ~candidate_wall, candidate_domain, topology.min_component_area)
    skeleton_original = skeleton_signature(original_wall)
    skeleton_candidate = skeleton_signature(candidate_wall)
    core_geometry = bool(recall >= 0.995 and precision >= 0.995) if geometric_core else True
    return ValidationResult(
        jacobian=bool(jacobian_min > 0.60), wall_components=oc == cc,
        free_space_components=ofc == cfc, euler=oe == ce, no_crop=no_crop,
        wall_core_integrity=recall >= 0.985 and bool(np.all((candidate_core > 0) <= (candidate_wall > 0))), jacobian_min=float(jacobian_min),
        original_wall_components=oc, candidate_wall_components=cc,
        original_free_components=ofc, candidate_free_components=cfc,
        original_euler=oe, candidate_euler=ce, wall_core_recall=recall,
        crop_recall=float(crop_recall), wall_core_precision=precision,
        wall_label_identity=True if wall_identity is None else wall_identity.passed,
        free_label_identity=True if free_identity is None else free_identity.passed,
        core_geometry=core_geometry,
        post_render_topology=True if post_render is None else post_render.passed,
        wall_identity=wall_identity, free_identity=free_identity,
        wall_seeds=wall_seeds, free_seeds=free_seeds,
        skeleton_original=skeleton_original, skeleton_candidate=skeleton_candidate,
        post_render=post_render)


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
    return [name for name, passed in validation.gate_checks().items() if not passed]


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
                    max_attempts: int=MAX_VARIANT_ATTEMPTS, save_success_debug: bool=True,
                    topology: Optional[TopologyReference]=None) -> VariantResult:
    p=LEVEL_PARAMS[level]; base_rng=np.random.default_rng(seed)
    amp=float(p["warp"]*wall_width)
    topology = topology or build_topology_reference(wall_mask, masks["wall_core"])
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
        expected_wall_labels=warp_label_map(topology.wall_labels,dx,dy,perspective_transform)
        expected_free_labels=warp_label_map(topology.free_labels,dx,dy,perspective_transform)
        expected_domain=warp_label_map(topology.domain_mask,dx,dy,perspective_transform)
        expected_core=warp_label_map(topology.wall_core,dx,dy,perspective_transform)
        core_bin=(wc>0).astype(np.uint8)
        structural=np.maximum((wm>0).astype(np.uint8), core_bin)
        edge_bin=np.maximum((we>0).astype(np.uint8),cv2.subtract(structural,core_bin))
        corridor_bin=cv2.subtract((cor>0).astype(np.uint8),structural)
        candidate_masks={"wall_mask":structural,"wall_core":core_bin,"wall_edge":edge_bin,"corridor_core":corridor_bin}
        crop_recall=calculate_crop_recall(wm_full,perspective_transform.crop_window)
        candidate_free=((expected_domain > 0) & ~(structural > 0)).astype(np.uint8)
        val=validator(wall_mask,candidate_masks["wall_mask"],masks["wall_core"],candidate_masks["wall_core"],
                      original_free=topology.domain_mask & ~(wall_mask > 0), candidate_free=candidate_free,
                      jacobian_min=jmin,crop_recall=crop_recall, topology=topology,
                      expected_wall_labels=expected_wall_labels, expected_free_labels=expected_free_labels,
                      expected_domain=expected_domain, expected_core=expected_core)
        val.jacobian = jac_ok
        metadata=dict(attempt=attempt, scale=scale, amp_px=amp*scale, jacobian_min=jmin,
            perspective_requested=perspective_requested, perspective_used=perspective_used,
            src_corners=perspective_transform.src.tolist(), dst_corners=perspective_transform.dst.tolist(),
            H=perspective_transform.H.tolist(), pad=perspective_transform.pad,
            crop_origin=list(perspective_transform.crop_origin), crop_x=perspective_transform.crop_window[0],
            crop_y=perspective_transform.crop_window[1], crop_width=perspective_transform.crop_window[2],
            crop_height=perspective_transform.crop_window[3], crop_recall=crop_recall)
        last_payload=(warped,val,metadata,candidate_masks,dx,dy,jac,mm)
        reasons=validation_failure_reasons(val)
        if not val.passed:
            attempts.append(AttemptResult(attempt, scale, amp*scale, perspective_used, jmin, val, reasons, metadata))
            continue
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
        post=validate_post_render(out, topology, expected_wall_labels, expected_free_labels,
                                  expected_domain, candidate_masks, mm, distractors)
        val.post_render=post
        val.post_render_topology=post.passed
        reasons=validation_failure_reasons(val)
        attempts.append(AttemptResult(attempt, scale, amp*scale, perspective_used, jmin, val, reasons, metadata))
        last_payload=(warped,val,metadata,candidate_masks,dx,dy,jac,mm)
        if not val.passed:
            continue
        if debug_dir is not None and save_success_debug:
            debug_dir.mkdir(parents=True,exist_ok=True)
            cv2.imwrite(str(debug_dir/f"displacement_{level}.png"),np.clip(np.dstack([_normalize_field(dx),_normalize_field(dy),np.zeros_like(dx)])*127.5+127.5,0,255).astype(np.uint8))
            cv2.imwrite(str(debug_dir/f"jacobian_{level}.png"),cv2.normalize(jac,None,0,255,cv2.NORM_MINMAX).astype(np.uint8))
            cv2.imwrite(str(debug_dir/f"wall_mask_{level}.png"),candidate_masks["wall_mask"]*255)
            cv2.imwrite(str(debug_dir/f"wall_core_{level}.png"),candidate_masks["wall_core"]*255)
            cv2.imwrite(str(debug_dir/f"corridor_core_{level}.png"),candidate_masks["corridor_core"]*255)
            cv2.imwrite(str(debug_dir/f"distractors_{level}.png"),distractors*255)
            render_gray=cv2.cvtColor(out,cv2.COLOR_RGB2GRAY)
            render_wall=(render_gray <= post.threshold).astype(np.uint8)
            render_wall[mm > 0]=0
            render_wall[distractors > 0]=0
            save_topology_debug(debug_dir, level, expected_wall_labels, candidate_masks["wall_mask"],
                                expected_free_labels, candidate_free,
                                render_wall, val.wall_identity, val.free_identity, post)
        metadata["wall_core_recall"]=val.wall_core_recall
        metadata["wall_core_precision"]=val.wall_core_precision
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
    domain=np.zeros((h,w),np.uint8); x,y,rw,rh=roi; domain[y:y+rh,x:x+rw]=1
    topology=build_topology_reference(wall,masks["wall_core"],domain)
    analysis=ImageAnalysis(w,h,rgb.shape[2],roi,otsu,wall_width,float((wall>0).mean()),float((cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)[...,1]>85).mean()),int(marker.sum()))
    _save_rgb(debug/"wall_mask_original.png",np.repeat((wall*255)[...,None],3,axis=2)); _save_rgb(debug/"wall_core_original.png",np.repeat((masks["wall_core"]*255)[...,None],3,axis=2)); _save_rgb(debug/"corridor_core_original.png",np.repeat((masks["corridor_core"]*255)[...,None],3,axis=2))
    variants=[]; validations={}; attempts={}; results={}; failed_levels=[]
    original_for_sheet=rgb.copy()
    for idx,level in enumerate(levels,1):
        result=generate_variant(rgb,wall,masks,marker,wall_width,level,args.seed+idx*7919,args.no_perspective,debug,save_success_debug=args.debug,topology=topology)
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
