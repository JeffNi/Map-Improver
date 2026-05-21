"""
enhance_map.py
--------------
Polished flat-style map enhancement pipeline.

Stages:
  1. Load & classify pixels into biomes
  2. Edge smoothing (blur + re-snap to palette)
  3. Ocean depth gradient + shallow water
  4. Terrain inner-edge shading
  5. Biome texture overlays (Perlin noise)
  6. Composite & save
"""

import os
import sys
import math
import numpy as np
from PIL import Image, ImageFilter
from scipy.ndimage import distance_transform_edt

try:
    import opensimplex
    HAS_NOISE = True
except ImportError:
    HAS_NOISE = False
    print("[warn] 'opensimplex' package not found — using numpy random noise instead.")

import config

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_image(path: str) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.array(img, dtype=np.uint8)


def save_image(arr: np.ndarray, path: str) -> None:
    Image.fromarray(arr.astype(np.uint8)).save(path)
    print(f"[✓] Saved → {path}")


def palette_from_config() -> tuple[np.ndarray, list[str]]:
    """Return (N×3 palette array, list of biome names) from config."""
    colors = list(config.BIOME_COLORS.keys())
    names  = list(config.BIOME_COLORS.values())
    return np.array(colors, dtype=np.float32), names


def classify_pixels(img: np.ndarray, palette: np.ndarray, names: list[str]) -> np.ndarray:
    """
    Assign each pixel the index of its nearest palette entry.
    Returns an int32 array of shape (H, W).
    """
    H, W, _ = img.shape
    flat = img.reshape(-1, 3).astype(np.float32)          # (H*W, 3)
    dists = np.sum((flat[:, None, :] - palette[None, :, :]) ** 2, axis=2)  # (H*W, N)
    labels = np.argmin(dists, axis=1).astype(np.int32)
    return labels.reshape(H, W)


def biome_mask(labels: np.ndarray, names: list[str], biome: str) -> np.ndarray:
    """Boolean mask for a single biome."""
    idx = names.index(biome)
    return labels == idx


def blend(base: np.ndarray, overlay: np.ndarray, alpha: float) -> np.ndarray:
    """Alpha-blend overlay onto base. alpha in [0,1]. Returns uint8."""
    return np.clip(base.astype(np.float32) * (1 - alpha) +
                   overlay.astype(np.float32) * alpha, 0, 255).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 – Edge smoothing
# ─────────────────────────────────────────────────────────────────────────────

def smooth_edges(img: np.ndarray, palette: np.ndarray, names: list[str],
                 radius: int) -> np.ndarray:
    """
    Blur the image slightly then re-snap each pixel to the nearest palette color.
    This rounds hard MS-Paint pixel staircases without blurring biome fills.
    """
    pil = Image.fromarray(img)
    blurred = np.array(pil.filter(ImageFilter.GaussianBlur(radius=radius)),
                       dtype=np.uint8)
    labels = classify_pixels(blurred, palette, names)
    # Reconstruct from palette
    H, W = labels.shape
    out = palette[labels.reshape(-1)].reshape(H, W, 3).astype(np.uint8)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 – Ocean depth + shallow water
# ─────────────────────────────────────────────────────────────────────────────

OCEAN_DEEP    = np.array([10,  60, 130], dtype=np.float32)   # deep ocean
OCEAN_MID     = np.array([30, 120, 200], dtype=np.float32)   # mid ocean
OCEAN_SHALLOW = np.array([60, 190, 160], dtype=np.float32)   # shallow turquoise

def ocean_depth(img: np.ndarray, labels: np.ndarray, names: list[str]) -> np.ndarray:
    """Replace flat ocean pixels with a depth gradient based on distance from land."""
    if "ocean" not in names:
        return img

    ocean_mask = biome_mask(labels, names, "ocean")
    land_mask  = ~ocean_mask

    # Distance from each ocean pixel to the nearest land pixel
    dist = distance_transform_edt(ocean_mask)   # 0 at land edge, grows into ocean

    r = config.SHALLOW_WATER_RADIUS
    out = img.copy().astype(np.float32)

    # Shallow band  [0, r]  → mix OCEAN_SHALLOW → OCEAN_MID
    shallow_band = ocean_mask & (dist <= r)
    if shallow_band.any():
        t = (dist[shallow_band] / r).clip(0, 1)          # 0 = coast, 1 = mid
        c = (1 - t)[:, None] * OCEAN_SHALLOW + t[:, None] * OCEAN_MID
        out[shallow_band] = c

    # Deep band  [r, r*4]  → mix OCEAN_MID → OCEAN_DEEP
    deep_band = ocean_mask & (dist > r)
    if deep_band.any():
        t = ((dist[deep_band] - r) / (r * 3)).clip(0, 1)
        c = (1 - t)[:, None] * OCEAN_MID + t[:, None] * OCEAN_DEEP
        out[deep_band] = c

    return out.astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3 – Terrain inner-edge shading
# ─────────────────────────────────────────────────────────────────────────────

def terrain_shading(img: np.ndarray, labels: np.ndarray, names: list[str]) -> np.ndarray:
    """
    Darken land pixels near biome boundaries (inner glow / coastal shadow).
    """
    out = img.copy().astype(np.float32)
    H, W, _ = img.shape

    # Build a mask: True where pixel is at a biome boundary
    # Shift labels by 1 in each direction and compare
    boundary = np.zeros((H, W), dtype=bool)
    boundary[:-1, :] |= (labels[:-1, :] != labels[1:, :])
    boundary[1:,  :] |= (labels[1:,  :] != labels[:-1, :])
    boundary[:, :-1] |= (labels[:, :-1] != labels[:, 1:])
    boundary[:, 1:]  |= (labels[:, 1:]  != labels[:, :-1])

    # Distance from each land pixel to nearest boundary
    land_mask = ~biome_mask(labels, names, "ocean") if "ocean" in names else np.ones((H, W), dtype=bool)
    dist_to_boundary = distance_transform_edt(~boundary)

    r = config.TERRAIN_SHADE_RADIUS
    shade_band = land_mask & (dist_to_boundary <= r)

    if shade_band.any():
        # t=0 at boundary (darkest), t=1 at r pixels in (no effect)
        t = (dist_to_boundary[shade_band] / r).clip(0, 1)
        alpha = (1 - t) * config.TERRAIN_SHADE_STRENGTH
        out[shade_band] = out[shade_band] * (1 - alpha[:, None])

    return out.astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# Stage 4 – Biome texture overlays
# ─────────────────────────────────────────────────────────────────────────────

def perlin_layer(H: int, W: int, scale: float, seed: int = 0) -> np.ndarray:
    """Generate a [0,1] float32 noise array of shape (H,W)."""
    if HAS_NOISE:
        opensimplex.seed(seed * 137 + 42)
        xs = np.linspace(0, scale, W, dtype=np.float64)
        ys = np.linspace(0, scale, H, dtype=np.float64)
        arr = opensimplex.noise2array(xs, ys).astype(np.float32)
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    else:
        rng = np.random.default_rng(seed)
        arr = rng.random((H, W), dtype=np.float32)
        # Simple multi-octave approximation with downsampling
        from PIL import Image as _Img
        base = _Img.fromarray((arr * 255).astype(np.uint8))
        blurred = base.filter(ImageFilter.GaussianBlur(radius=max(1, int(W / (scale * 8)))))
        arr = np.array(blurred, dtype=np.float32) / 255.0

    return arr


def texture_overlays(img: np.ndarray, labels: np.ndarray, names: list[str]) -> np.ndarray:
    """Blend per-biome Perlin noise over the image."""
    H, W, _ = img.shape
    out = img.copy().astype(np.float32)

    for i, biome in enumerate(names):
        opacity = config.TEXTURE_OPACITY.get(biome, 0.10)
        if opacity <= 0:
            continue
        scale = config.TEXTURE_SCALE.get(biome, 5.0)
        mask = labels == i
        if not mask.any():
            continue

        print(f"  Generating texture for '{biome}'...")
        noise = perlin_layer(H, W, scale, seed=i)

        # Convert noise to a darkening/lightening value centered on 0
        noise_centered = (noise - 0.5) * 2   # [-1, 1]
        # Apply: positive noise slightly lightens, negative slightly darkens
        delta = noise_centered[:, :, None] * opacity * 80  # max ±80*opacity shift
        out[mask] = np.clip(out[mask] + delta[mask], 0, 255)

    return out.astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    in_path  = os.path.join(script_dir, config.INPUT_PATH)
    out_path = os.path.join(script_dir, config.OUTPUT_PATH)

    print(f"[1/7] Loading '{in_path}' ...")
    img = load_image(in_path)
    H, W, _ = img.shape
    print(f"      {W}×{H} px")

    palette, names = palette_from_config()
    print(f"[2/7] Classifying {len(names)} biomes ...")
    labels = classify_pixels(img, palette, names)

    biome_counts = {names[i]: int((labels == i).sum()) for i in range(len(names))}
    for b, c in sorted(biome_counts.items(), key=lambda x: -x[1]):
        if c > 0:
            print(f"      {b}: {c:,} px")

    print("[3/7] Smoothing edges ...")
    img = smooth_edges(img, palette, names, radius=config.SMOOTH_BLUR_RADIUS)
    labels = classify_pixels(img, palette, names)   # re-classify after smooth

    print("[4/7] Ocean depth gradient ...")
    img = ocean_depth(img, labels, names)

    print("[5/7] Terrain edge shading ...")
    img = terrain_shading(img, labels, names)

    print("[6/7] Biome texture overlays ...")
    img = texture_overlays(img, labels, names)

    print("[7/7] Saving ...")
    save_image(img, out_path)


if __name__ == "__main__":
    main()
