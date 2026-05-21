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

# Edge-smoothing blur radius (pixels). Higher = rounder edges but more blur.
SMOOTH_BLUR_RADIUS = 2

# Input / output paths (relative to the script location)
INPUT_PATH  = "Maps/Map Geography.png"
OUTPUT_PATH = "Maps/Map Geography Enhanced.png"
