from os.path import join

cloth_channels = 19
CROP_BOUNDS = (
    (160, 672),  # HEIGHT (determined by looking at sample frames. want to get the head)
    (14, 526),  # WIDTH (calculated by (540 - 512)/2, then centered  )
)

ROOT = "data"

ANDREW_TEXTURE = join(ROOT, "andrew/texture")
ANDREW_ROIS = join(ROOT, "andrew/rois.csv")
ANDREW_CLOTHING_SEG = join(ROOT, "andrew/clothing")
ANDREW_BODY_SEG = join(ROOT, "andrew/body")
ANDREW_BODY_SEG_MEAN = [0.0265, 0.041, 0.034]
ANDREW_BODY_SEG_STD = [0.110, 0.137, 0.127]

TINA_TEXTURE = join(ROOT, "tina/texture")
TINA_ROIS = join(ROOT, "tina/rois.csv")
TINA_CLOTHING_SEG = join(ROOT, "tina/clothing")
TINA_BODY_SEG = join(ROOT, "tina/body")
TINA_BODY_SEG_MEAN = [0.016, 0.024, 0.018]
TINA_BODY_SEG_STD = [0.087, 0.107, 0.092]
