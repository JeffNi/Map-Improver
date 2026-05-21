# ─────────────────────────────────────────────────────────────────────────────
# Biome color mapping
# Paste the exact hex code for each biome (copy from Paint / any color picker).
# Format: "#RRGGBB"  e.g. "#00AAFF"
# ─────────────────────────────────────────────────────────────────────────────

def _h(hex_str: str) -> tuple[int, int, int]:
    h = hex_str.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))

BIOME_COLORS: dict[tuple[int, int, int], str] = {
    _h("#00A2E8"): "ocean",
    _h("#00FF0B"): "jungle",
    _h("#22B14C"): "forest",
    _h("#35CF10"): "plains",
    _h("#FFC90E"): "desert",
    _h("#FFF200"): "beach",
    _h("#FFFFFF"): "tundra",
    _h("#7F7F7F"): "mountain",
    _h("#DBDBDB"): "stony_shore",
    _h("#476E1C"): "swamp",
    _h("#C3C3C3"): "shattered_lands",
}

# ─────────────────────────────────────────────────────────────────────────────
# Enhancement tuning — adjust these to taste after a first run
# ─────────────────────────────────────────────────────────────────────────────

# How many pixels out from the coast the shallow water extends
SHALLOW_WATER_RADIUS = 30

# Strength of the inner-edge shading on land biomes (0.0 – 1.0)
# Higher = darker edges
TERRAIN_SHADE_STRENGTH = 0.30

# How far inland the edge shading reaches (pixels)
TERRAIN_SHADE_RADIUS = 18

# Texture overlay opacity per biome (0.0 = none, 1.0 = full)
# Values around 0.08–0.18 look natural
TEXTURE_OPACITY: dict[str, float] = {
    "jungle":          0.12,
    "forest":          0.12,
    "plains":          0.10,
    "desert":          0.15,
    "beach":           0.08,
    "tundra":          0.10,
    "mountain":        0.18,
    "stony_shore":     0.14,
    "swamp":           0.12,
    "shattered_lands": 0.14,
    "ocean":           0.06,
}

# Perlin noise scale per biome (higher = finer grain)
TEXTURE_SCALE: dict[str, float] = {
    "jungle":          6.0,
    "forest":          5.0,
    "plains":          4.0,
    "desert":          3.0,
    "beach":           4.0,
    "tundra":          3.5,
    "mountain":        7.0,
    "stony_shore":     6.0,
    "swamp":           4.5,
    "shattered_lands": 5.0,
    "ocean":           2.5,
}

# ─────────────────────────────────────────────────────────────────────────────
# Jungle foliage stipple
# ─────────────────────────────────────────────────────────────────────────────

# Shift the base jungle color toward this richer dark green (away from neon)
# Format: (R, G, B) — blend target
JUNGLE_BASE_COLOR = (18, 140, 28)

# How strongly to push pixels toward JUNGLE_BASE_COLOR (0=keep neon, 1=full recolor)
JUNGLE_BASE_STRENGTH = 0.70

# Noise scales for the three octaves (pixels) — large→canopy blobs, fine→stipple
JUNGLE_SCALE_LARGE  = 70.0
JUNGLE_SCALE_MEDIUM = 22.0
JUNGLE_SCALE_FINE   = 7.0

# Mix weights for each octave (should sum to ~1.0)
JUNGLE_MIX_LARGE  = 0.35
JUNGLE_MIX_MEDIUM = 0.40
JUNGLE_MIX_FINE   = 0.25

# Overall stipple brightness range (±pixels around the recoloured base)
JUNGLE_STIPPLE_RANGE = 55

# Tropical flower colour tint
JUNGLE_FLOWER_COLOR   = (220, 90, 130)   # warm coral-pink (R, G, B)
JUNGLE_FLOWER_SCALE   = 20.0             # noise scale — smaller = tighter flower clusters
JUNGLE_FLOWER_DENSITY = 0.40             # 0–1, fraction of jungle that has flowers
JUNGLE_FLOWER_STRENGTH = 135            # max colour shift toward flower tint

# ─────────────────────────────────────────────────────────────────────────────
# Forest canopy texture  (same stipple approach as jungle, tuned for temperate)
# ─────────────────────────────────────────────────────────────────────────────

# Push forest pixels toward this darker, richer green
FOREST_BASE_COLOR    = (22, 110, 35)
FOREST_BASE_STRENGTH = 0.55

# Noise octave scales — larger blobs than jungle (older, denser canopy)
FOREST_SCALE_LARGE  = 90.0
FOREST_SCALE_MEDIUM = 32.0
FOREST_SCALE_FINE   = 10.0

# Octave mix weights
FOREST_MIX_LARGE  = 0.45
FOREST_MIX_MEDIUM = 0.35
FOREST_MIX_FINE   = 0.20

# Brightness variation range (±pixels) — slightly less contrast than jungle
FOREST_STIPPLE_RANGE = 42

# ─────────────────────────────────────────────────────────────────────────────
# Plains rolling undulation
# ─────────────────────────────────────────────────────────────────────────────

PLAINS_BASE_COLOR    = (42, 175, 22)   # slightly warmer meadow green
PLAINS_BASE_STRENGTH = 0.60            # shift enough to give contrast room

PLAINS_SCALE_LARGE  = 120.0           # very broad rolling hills
PLAINS_SCALE_MEDIUM =  45.0           # gentler secondary variation

PLAINS_MIX_LARGE  = 0.70
PLAINS_MIX_MEDIUM = 0.30

PLAINS_UNDULATION_RANGE = 45          # ±pixel brightness — intentionally flatter than forest

# ─────────────────────────────────────────────────────────────────────────────
# Mountain hillshading + rock texture
# ─────────────────────────────────────────────────────────────────────────────

# Blend mountain pixels toward this base rock colour before shading
MOUNTAIN_BASE_COLOR    = (148, 140, 128)
MOUNTAIN_BASE_STRENGTH = 0.45

# Light source direction (standard cartographic NW illumination)
MOUNTAIN_LIGHT_AZIMUTH  = 315   # degrees clockwise from north
MOUNTAIN_LIGHT_ALTITUDE = 45    # degrees above horizon

# Vertical exaggeration applied to the height field — higher = more dramatic shadows
MOUNTAIN_HEIGHT_SCALE = 5.0

# Shadow darkness: 0 = pure black in shadow, 1 = no shading at all
MOUNTAIN_AMBIENT = 0.28

# Fine rock surface texture
MOUNTAIN_ROCK_SCALE    = 14.0   # noise scale (pixels)
MOUNTAIN_ROCK_STRENGTH = 30     # ±pixel brightness variation

# ─────────────────────────────────────────────────────────────────────────────
# Desert dune texture
# ─────────────────────────────────────────────────────────────────────────────

# Angle the dune ridges run at (degrees from horizontal)
DUNE_ANGLE = 30

# Distance between dune crests in pixels
DUNE_WAVELENGTH = 55

# How much the ridges are warped by noise (higher = more wavy/irregular)
DUNE_WARP_STRENGTH = 25.0

# Opacity of the dune overlay (0.0–1.0). Higher = more visible ridges.
DUNE_OPACITY = 0.15

# ─────────────────────────────────────────────────────────────────────────────
# Shattered Lands salt flat texture
# ─────────────────────────────────────────────────────────────────────────────

# Distance between primary hex cell centers in pixels
SALTFLAT_CELL_SIZE = 65

# How much to randomly offset each hex center (0.0 = perfect grid, 0.5 = chaotic)
SALTFLAT_JITTER = 0.42

# Secondary (smaller) crack layer size as a fraction of the primary
SALTFLAT_SECONDARY_SCALE = 0.32

# How much the secondary crack layer blends in (0.0 = none)
SALTFLAT_SECONDARY_MIX = 0.30

# Interior grunge noise strength (0.0 = flat cells, 1.0 = very dirty)
SALTFLAT_GRUNGE = 0.40

# Softness of the crack edges in pixels (higher = fuzzier cracks)
SALTFLAT_CRACK_WIDTH = 3

# Overall biome brightness multiplier — lower = darker/more ominous area
SALTFLAT_BIOME_DARKNESS = 0.96

# How dark crack pixels get (0=no effect, 150=near black)
SALTFLAT_CRACK_DARKNESS = 115

# Surface variation strength within cell interiors (±pixels)
SALTFLAT_SURFACE_VAR = 35

# Dust colour tint applied to cell interiors: (delta_R, delta_G, delta_B)
# Positive shifts warm/yellow, negative shifts cool. Keep small (< 25).
SALTFLAT_DUST_TINT = (14, 9, -14)

# Shore brightening — pale whitish band around the biome edge
SALTFLAT_SHORE_RADIUS = 50       # pixels inward from biome boundary
SALTFLAT_SHORE_STRENGTH = 0.90   # 0=no effect, 1=pure white at edge

# Purple crack glow
SALTFLAT_GLOW_COLOR = (55, -15, 100)   # (R, G, B) delta — violet/purple
SALTFLAT_GLOW_FRACTION = 0.38          # fraction of map where cracks glow (noise threshold)
SALTFLAT_GLOW_STRENGTH = 1.8             # max brightness of glow (pixel units)
SALTFLAT_GLOW_BLUR = 5                  # Gaussian blur radius for glow halo (pixels)

# ─────────────────────────────────────────────────────────────────────────────
# Edge-smoothing blur radius (pixels). Higher = rounder edges but more blur.
SMOOTH_BLUR_RADIUS = 2

# Input / output paths (relative to the script location)
INPUT_PATH  = "Maps/Map Geography.png"
OUTPUT_PATH = "Maps/Map Geography Enhanced.png"
