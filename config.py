ANDREW_CLOTHING_SEG = (
    "/run/media/espina/DATA/outfit-transfer/andrew_clothing_seg/probability_maps"
)


CROP_BOUNDS = (
    (160, 672),  # HEIGHT (determined by looking at sample frames. want to get the head)
    (14, 526),  # WIDTH (calculated by (540 - 512)/2, then centered  )
)


ANDREW_BODY_SEG = "/run/media/espina/DATA/outfit-transfer/andrew_body_seg"
# ANDREW_BODY_SEG_MEAN = [0.02548507578066196, 0.04090483438072337, 0.034430569821185406]
# ANDREW_BODY_SEG_STD = [0.11019536986470552, 0.137329593577946, 0.12687526036868005]
ANDREW_BODY_SEG_MEAN = [0.0265, 0.041, 0.034]
ANDREW_BODY_SEG_STD = [0.110, 0.137, 0.127]


TINA_CLOTHING_SEG = (
    "/run/media/espina/DATA/outfit-transfer/andrew_clothing_seg/probability_maps"
)


TINA_BODY_SEG = "/run/media/espina/DATA/outfit-transfer/tina_body_seg"
# TINA_BODY_SEG_MEAN = [0.015632749085877164, 0.024117475956897578, 0.01753022733575702]
# TINA_BODY_SEG_STD = [0.08719314278167126, 0.10735163948663556, 0.09215301293073803]
TINA_BODY_SEG_MEAN = [0.016, 0.024, 0.018]
TINA_BODY_SEG_STD = [0.087, 0.107, 0.092]
