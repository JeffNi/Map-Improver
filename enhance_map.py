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
from scipy.ndimage import distance_transform_edt, median_filter
from scipy.spatial import KDTree

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


def clean_labels(labels: np.ndarray, radius: int = 2) -> np.ndarray:
    """
    Remove isolated mis-classified pixels by median-filtering the label map.
    Any stray pixel surrounded by a different biome gets corrected.
    radius=2 → 5×5 neighbourhood; increase for larger spurious blobs.
    """
    return median_filter(labels, size=radius * 2 + 1).astype(np.int32)


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


def jungle_layer(H: int, W: int) -> np.ndarray:
    """
    Multi-scale foliage stipple texture for jungle.
    Returns a [0,1] float32 array: bright = sunlit canopy top, dark = shadow.
    Three octaves: large canopy blobs, medium clusters, fine leaf stipple.
    """
    def _octave(scale: float, seed: int) -> np.ndarray:
        if HAS_NOISE:
            opensimplex.seed(seed)
            xs = np.linspace(0, scale, W, dtype=np.float64)
            ys = np.linspace(0, scale, H, dtype=np.float64)
            n = opensimplex.noise2array(xs, ys).astype(np.float32)
        else:
            rng = np.random.default_rng(seed)
            raw = rng.random((H, W), dtype=np.float32)
            n = np.array(
                Image.fromarray((raw * 255).astype(np.uint8))
                     .filter(ImageFilter.GaussianBlur(radius=int(scale / 3))),
                dtype=np.float32
            ) / 127.5 - 1.0
        n = (n - n.min()) / (n.max() - n.min() + 1e-8)
        return n

    large  = _octave(H / config.JUNGLE_SCALE_LARGE,  seed=500)
    medium = _octave(H / config.JUNGLE_SCALE_MEDIUM, seed=501)
    fine   = _octave(H / config.JUNGLE_SCALE_FINE,   seed=502)

    arr = (large  * config.JUNGLE_MIX_LARGE
         + medium * config.JUNGLE_MIX_MEDIUM
         + fine   * config.JUNGLE_MIX_FINE)
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    return arr


def forest_layer(H: int, W: int) -> np.ndarray:
    """
    Multi-scale canopy stipple for forest — larger blobs than jungle,
    temperate palette, different seeds.
    """
    def _octave(scale: float, seed: int) -> np.ndarray:
        if HAS_NOISE:
            opensimplex.seed(seed)
            xs = np.linspace(0, scale, W, dtype=np.float64)
            ys = np.linspace(0, scale, H, dtype=np.float64)
            n = opensimplex.noise2array(xs, ys).astype(np.float32)
        else:
            rng = np.random.default_rng(seed)
            raw = rng.random((H, W), dtype=np.float32)
            n = np.array(
                Image.fromarray((raw * 255).astype(np.uint8))
                     .filter(ImageFilter.GaussianBlur(radius=int(scale / 3))),
                dtype=np.float32
            ) / 127.5 - 1.0
        n = (n - n.min()) / (n.max() - n.min() + 1e-8)
        return n

    large  = _octave(H / config.FOREST_SCALE_LARGE,  seed=510)
    medium = _octave(H / config.FOREST_SCALE_MEDIUM, seed=511)
    fine   = _octave(H / config.FOREST_SCALE_FINE,   seed=512)

    arr = (large  * config.FOREST_MIX_LARGE
         + medium * config.FOREST_MIX_MEDIUM
         + fine   * config.FOREST_MIX_FINE)
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    return arr


def plains_layer(H: int, W: int) -> np.ndarray:
    """
    Two-octave rolling undulation for plains — broad and low-contrast.
    Returns [0,1] float32 array.
    """
    def _octave(scale: float, seed: int) -> np.ndarray:
        if HAS_NOISE:
            opensimplex.seed(seed)
            xs = np.linspace(0, scale, W, dtype=np.float64)
            ys = np.linspace(0, scale, H, dtype=np.float64)
            n = opensimplex.noise2array(xs, ys).astype(np.float32)
        else:
            rng = np.random.default_rng(seed)
            raw = rng.random((H, W), dtype=np.float32)
            n = np.array(
                Image.fromarray((raw * 255).astype(np.uint8))
                     .filter(ImageFilter.GaussianBlur(radius=int(H / (scale * 4)))),
                dtype=np.float32
            ) / 127.5 - 1.0
        return (n - n.min()) / (n.max() - n.min() + 1e-8)

    large  = _octave(H / config.PLAINS_SCALE_LARGE,  seed=520)
    medium = _octave(H / config.PLAINS_SCALE_MEDIUM, seed=521)

    arr = large * config.PLAINS_MIX_LARGE + medium * config.PLAINS_MIX_MEDIUM
    return (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)


def mountain_layer(H: int, W: int, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (shade, rock) float32 arrays of shape (H, W).
      shade : 0=fully shadowed, 1=fully lit  (hillshading from NW light)
      rock  : [0,1] fine surface texture noise
    Height field = distance-from-edge * noise, so biome edges = low, interior = peaks.
    """
    # --- height field ---
    dist = distance_transform_edt(mask).astype(np.float32)
    dist_norm = dist / (dist.max() + 1e-8)

    if HAS_NOISE:
        opensimplex.seed(600)
        xs = np.linspace(0, H / 40.0, W, dtype=np.float64)
        ys = np.linspace(0, H / 40.0, H, dtype=np.float64)
        hn = opensimplex.noise2array(xs, ys).astype(np.float32)
        opensimplex.seed(601)
        xs2 = np.linspace(0, H / 15.0, W, dtype=np.float64)
        ys2 = np.linspace(0, H / 15.0, H, dtype=np.float64)
        hn2 = opensimplex.noise2array(xs2, ys2).astype(np.float32)
        height_noise = (hn * 0.65 + hn2 * 0.35)
    else:
        rng = np.random.default_rng(600)
        height_noise = rng.random((H, W), dtype=np.float32) * 2 - 1

    height_noise = (height_noise - height_noise.min()) / (height_noise.max() - height_noise.min() + 1e-8)
    height = dist_norm * (0.6 + height_noise * 0.4) * config.MOUNTAIN_HEIGHT_SCALE

    # --- hillshading via surface normals ---
    dzdy, dzdx = np.gradient(height)
    length = np.sqrt(dzdx ** 2 + dzdy ** 2 + 1.0)
    nx = -dzdx / length
    ny = -dzdy / length
    nz =  1.0   / length

    az  = math.radians(config.MOUNTAIN_LIGHT_AZIMUTH)
    alt = math.radians(config.MOUNTAIN_LIGHT_ALTITUDE)
    lx =  math.cos(alt) * math.sin(az)
    ly = -math.cos(alt) * math.cos(az)   # negative: image y is south-down
    lz =  math.sin(alt)

    shade = (nx * lx + ny * ly + nz * lz).clip(0, 1).astype(np.float32)

    # --- rock surface texture ---
    if HAS_NOISE:
        opensimplex.seed(602)
        rx = np.linspace(0, H / config.MOUNTAIN_ROCK_SCALE, W, dtype=np.float64)
        ry = np.linspace(0, H / config.MOUNTAIN_ROCK_SCALE, H, dtype=np.float64)
        rock = opensimplex.noise2array(rx, ry).astype(np.float32)
    else:
        rock = np.random.default_rng(602).random((H, W), dtype=np.float32) * 2 - 1
    rock = (rock - rock.min()) / (rock.max() - rock.min() + 1e-8)

    return shade, rock


def dune_layer(H: int, W: int) -> np.ndarray:
    """
    Generate a [0,1] float32 dune texture of shape (H,W).
    Uses rotated sine waves warped by low-frequency noise to simulate sand ridges.
    """
    angle_rad = math.radians(config.DUNE_ANGLE)
    cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)

    ys, xs = np.meshgrid(np.arange(H, dtype=np.float32),
                         np.arange(W, dtype=np.float32), indexing="ij")

    # Rotate coordinates so ridges run at DUNE_ANGLE
    proj = xs * cos_a + ys * sin_a  # distance along the dune-normal axis

    # Low-frequency warp noise (perturbs the ridge phase per pixel)
    if HAS_NOISE:
        opensimplex.seed(999)
        warp_xs = np.linspace(0, 2.5, W, dtype=np.float64)
        warp_ys = np.linspace(0, 2.5, H, dtype=np.float64)
        warp = opensimplex.noise2array(warp_xs, warp_ys).astype(np.float32)
    else:
        rng = np.random.default_rng(999)
        raw = rng.random((H, W), dtype=np.float32)
        warp = np.array(
            Image.fromarray((raw * 255).astype(np.uint8))
                 .filter(ImageFilter.GaussianBlur(radius=40)),
            dtype=np.float32
        ) / 255.0 * 2 - 1

    phase = proj + warp * config.DUNE_WARP_STRENGTH

    # Shaped sine: abs(sin) gives a smooth ridge-and-trough, then bias toward
    # sharp crests using a power curve (dunes are steep on one side)
    raw_wave = np.sin(2 * math.pi * phase / config.DUNE_WAVELENGTH)
    # Asymmetric shaping: compress troughs, sharpen crests
    shaped = (raw_wave + 1) / 2          # [0, 1]
    shaped = shaped ** 0.6               # bias toward crests (power < 1 → push up)

    # Second finer octave for micro-ripples
    if HAS_NOISE:
        opensimplex.seed(998)
        ripple_xs = np.linspace(0, 5.0, W, dtype=np.float64)
        ripple_ys = np.linspace(0, 5.0, H, dtype=np.float64)
        ripple = opensimplex.noise2array(ripple_xs, ripple_ys).astype(np.float32)
        ripple = (ripple - ripple.min()) / (ripple.max() - ripple.min() + 1e-8)
    else:
        ripple = np.zeros_like(shaped)

    arr = shaped * 0.80 + ripple * 0.20
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    return arr


def _hex_voronoi_field(H: int, W: int, cell_size: float,
                       jitter_frac: float, seed: int) -> np.ndarray:
    """Jittered hexagonal Voronoi field, returns [0,1] crack map (0=crack, 1=interior)."""
    rng = np.random.default_rng(seed)
    s = cell_size
    jitter = jitter_frac * s
    row_h = s * math.sqrt(3) / 2
    seed_list = []
    row = 0
    y = -s
    while y < H + s:
        x_off = (s / 2) if (row % 2 == 1) else 0.0
        x = -s + x_off
        while x < W + s:
            seed_list.append([x + rng.uniform(-jitter, jitter),
                               y + rng.uniform(-jitter, jitter)])
            x += s
        y += row_h
        row += 1

    seeds = np.array(seed_list, dtype=np.float32)
    py, px = np.meshgrid(np.arange(H, dtype=np.float32),
                         np.arange(W, dtype=np.float32), indexing="ij")
    pixel_coords = np.column_stack([px.ravel(), py.ravel()])
    tree = KDTree(seeds)
    dists, _ = tree.query(pixel_coords, k=2)
    d1 = dists[:, 0].reshape(H, W).astype(np.float32)
    d2 = dists[:, 1].reshape(H, W).astype(np.float32)
    boundary_dist = (d2 - d1).clip(0, None)
    crack_w = float(config.SALTFLAT_CRACK_WIDTH)
    return np.tanh(boundary_dist / crack_w).astype(np.float32)


def saltflat_layer(H: int, W: int) -> np.ndarray:
    """
    Multi-scale jittered hex Voronoi crack map with grungy interiors.
    Returns a [0,1] array: 0 = deep crack, 1 = dirty cell interior.
    """
    s = float(config.SALTFLAT_CELL_SIZE)
    j = config.SALTFLAT_JITTER

    # Primary large cracks
    primary = _hex_voronoi_field(H, W, s, j, seed=42)

    # Secondary finer cracks (run inside primary cells)
    secondary = _hex_voronoi_field(H, W, s * config.SALTFLAT_SECONDARY_SCALE,
                                   min(j + 0.08, 0.55), seed=43)

    mix = config.SALTFLAT_SECONDARY_MIX
    cracks = primary * (1 - mix) + secondary * mix

    # Interior grunge — high-frequency noise to break up the flat surfaces
    if HAS_NOISE:
        opensimplex.seed(77)
        xs_1d = np.linspace(0, 8.0, W, dtype=np.float64)
        ys_1d = np.linspace(0, 8.0, H, dtype=np.float64)
        grunge = opensimplex.noise2array(xs_1d, ys_1d).astype(np.float32)
        grunge = (grunge - grunge.min()) / (grunge.max() - grunge.min() + 1e-8)
    else:
        rng = np.random.default_rng(77)
        grunge = rng.random((H, W), dtype=np.float32)

    g = config.SALTFLAT_GRUNGE
    arr = cracks * (1 - g) + grunge * g

    # Re-normalise but keep cracks near 0 by anchoring the minimum
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    return arr


def texture_overlays(img: np.ndarray, labels: np.ndarray, names: list[str]) -> np.ndarray:
    """Blend per-biome Perlin noise over the image."""
    H, W, _ = img.shape
    out = img.copy().astype(np.float32)

    for i, biome in enumerate(names):
        mask = labels == i
        if not mask.any():
            continue

        print(f"  Generating texture for '{biome}'...")

        if biome == "jungle":
            noise = jungle_layer(H, W)
            # 1. Recolor base away from neon toward rich dark green
            base_target = np.array(config.JUNGLE_BASE_COLOR, dtype=np.float32)
            s = config.JUNGLE_BASE_STRENGTH
            out[mask] = out[mask] * (1 - s) + base_target[None, :] * s
            # 2. Apply multi-scale stipple as brightness variation
            stipple = (noise[mask] - 0.5) * config.JUNGLE_STIPPLE_RANGE
            out[mask] += stipple[:, None]
            # 3. Sparse flower tint — pink patches where flower_noise > threshold
            if HAS_NOISE:
                opensimplex.seed(503)
                fx = np.linspace(0, H / config.JUNGLE_FLOWER_SCALE, W, dtype=np.float64)
                fy = np.linspace(0, H / config.JUNGLE_FLOWER_SCALE, H, dtype=np.float64)
                flower_noise = opensimplex.noise2array(fx, fy).astype(np.float32)
                flower_noise = (flower_noise - flower_noise.min()) / (flower_noise.max() - flower_noise.min() + 1e-8)
            else:
                flower_noise = np.random.default_rng(503).random((H, W), dtype=np.float32)
            # Blur noise before thresholding — creates soft, feathered patch edges
            flower_pil = Image.fromarray((flower_noise * 255).astype(np.uint8))
            flower_noise = np.array(
                flower_pil.filter(ImageFilter.GaussianBlur(radius=4)),
                dtype=np.float32
            ) / 255.0
            flower_noise = (flower_noise - flower_noise.min()) / (flower_noise.max() - flower_noise.min() + 1e-8)
            threshold = 1.0 - config.JUNGLE_FLOWER_DENSITY
            flower_mask = mask & (flower_noise > threshold)
            if flower_mask.any():
                intensity = (flower_noise[flower_mask] - threshold) / (1.0 - threshold + 1e-8)
                # Additive delta: push R and B up, pull G down → reads as pink on any green base
                fr, fg, fb = config.JUNGLE_FLOWER_COLOR
                flower_delta = np.array([fr - 30, fg - 160, fb - 10], dtype=np.float32)
                out[flower_mask] += (intensity * config.JUNGLE_FLOWER_STRENGTH / 100.0)[:, None] * flower_delta[None, :]
            out[mask] = np.clip(out[mask], 0, 255)
            continue
        elif biome == "forest":
            noise = forest_layer(H, W)
            # 1. Recolor base toward darker temperate green
            base_target = np.array(config.FOREST_BASE_COLOR, dtype=np.float32)
            s = config.FOREST_BASE_STRENGTH
            out[mask] = out[mask] * (1 - s) + base_target[None, :] * s
            # 2. Canopy stipple — broader, calmer variation than jungle
            stipple = (noise[mask] - 0.5) * config.FOREST_STIPPLE_RANGE
            out[mask] += stipple[:, None]
            out[mask] = np.clip(out[mask], 0, 255)
            continue
        elif biome == "plains":
            noise = plains_layer(H, W)
            base_target = np.array(config.PLAINS_BASE_COLOR, dtype=np.float32)
            s = config.PLAINS_BASE_STRENGTH
            out[mask] = out[mask] * (1 - s) + base_target[None, :] * s
            undulation = (noise[mask] - 0.5) * config.PLAINS_UNDULATION_RANGE
            out[mask] += undulation[:, None]
            out[mask] = np.clip(out[mask], 0, 255)
            continue
        elif biome == "mountain":
            shade, rock = mountain_layer(H, W, mask)
            # 1. Shift base toward warm grey rock colour
            base_target = np.array(config.MOUNTAIN_BASE_COLOR, dtype=np.float32)
            s = config.MOUNTAIN_BASE_STRENGTH
            out[mask] = out[mask] * (1 - s) + base_target[None, :] * s
            # 2. Fine rock surface variation
            rock_delta = (rock[mask] - 0.5) * config.MOUNTAIN_ROCK_STRENGTH
            out[mask] += rock_delta[:, None]
            # 3. Hillshading: ambient + diffuse
            light = config.MOUNTAIN_AMBIENT + (1.0 - config.MOUNTAIN_AMBIENT) * shade[mask]
            out[mask] *= light[:, None]
            out[mask] = np.clip(out[mask], 0, 255)
            continue
        elif biome == "desert":
            noise = dune_layer(H, W)
            opacity = config.DUNE_OPACITY
        elif biome == "shattered_lands":
            noise = saltflat_layer(H, W)
            # Shore distance — used to fade ALL texture effects to zero at the coast
            shore_dist = distance_transform_edt(mask)
            shore_r = float(config.SALTFLAT_SHORE_RADIUS)
            interior_factor = np.clip(shore_dist / shore_r, 0, 1)  # 0=edge, 1=deep interior

            # 1. Very slight overall darkening (keeps the area mostly white/light)
            out[mask] *= config.SALTFLAT_BIOME_DARKNESS

            # 2. Deep black cracks — faded out completely near shore
            crack_depth_full = (1.0 - noise) ** 2                   # (H,W) 0=interior 1=crack
            effective_crack = crack_depth_full[mask] * interior_factor[mask]
            out[mask] -= effective_crack[:, None] * config.SALTFLAT_CRACK_DARKNESS

            # 3. Surface variation — also faded at shore
            surface = (noise[mask] - 0.5) * interior_factor[mask] * config.SALTFLAT_SURFACE_VAR
            out[mask] += surface[:, None]

            # 4. Dust tint on cell interiors, faded at shore
            int_blend = np.clip((noise[mask] - 0.3) / 0.7, 0, 1) * interior_factor[mask]
            tint = np.array(config.SALTFLAT_DUST_TINT, dtype=np.float32)
            out[mask] += int_blend[:, None] * tint[None, :]

            # 5. Shore → pure white fade (no texture)
            shore_band = mask & (shore_dist <= shore_r)
            if shore_band.any():
                t = (1.0 - shore_dist[shore_band] / shore_r).clip(0, 1)
                white = np.array([248, 245, 240], dtype=np.float32)
                s = (config.SALTFLAT_SHORE_STRENGTH * t)[:, None]
                out[shore_band] = out[shore_band] * (1 - s) + white[None, :] * s

            # 6. Purple glow — blur crack lines into a halo, then select a subset with noise
            crack_bright = (crack_depth_full * interior_factor).astype(np.float32)
            crack_pil = Image.fromarray((crack_bright * 255).astype(np.uint8))
            glow_halo = np.array(
                crack_pil.filter(ImageFilter.GaussianBlur(radius=config.SALTFLAT_GLOW_BLUR)),
                dtype=np.float32
            ) / 255.0
            if HAS_NOISE:
                opensimplex.seed(333)
                gx = np.linspace(0, 4.0, W, dtype=np.float64)
                gy = np.linspace(0, 4.0, H, dtype=np.float64)
                glow_sel = opensimplex.noise2array(gx, gy).astype(np.float32)
                glow_sel = (glow_sel - glow_sel.min()) / (glow_sel.max() - glow_sel.min() + 1e-8)
            else:
                glow_sel = np.random.default_rng(333).random((H, W), dtype=np.float32)
            threshold = 1.0 - config.SALTFLAT_GLOW_FRACTION
            glow_map = glow_halo * np.clip((glow_sel - threshold) / (1 - threshold + 1e-8), 0, 1)
            glow_rgb = np.array(config.SALTFLAT_GLOW_COLOR, dtype=np.float32)
            out[mask] += (glow_map[mask] * config.SALTFLAT_GLOW_STRENGTH)[:, None] * glow_rgb[None, :]

            out[mask] = np.clip(out[mask], 0, 255)
            continue  # skip generic noise blend below
        else:
            opacity = config.TEXTURE_OPACITY.get(biome, 0.10)
            if opacity <= 0:
                continue
            scale = config.TEXTURE_SCALE.get(biome, 5.0)
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
    labels = clean_labels(labels)   # remove stray mis-coloured pixels

    biome_counts = {names[i]: int((labels == i).sum()) for i in range(len(names))}
    for b, c in sorted(biome_counts.items(), key=lambda x: -x[1]):
        if c > 0:
            print(f"      {b}: {c:,} px")

    print("[3/7] Smoothing edges ...")
    img = smooth_edges(img, palette, names, radius=config.SMOOTH_BLUR_RADIUS)
    # labels stay locked to the original classification — never re-classify after this point

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
