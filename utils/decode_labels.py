import numpy as np
from PIL import Image

N_CLOTH_CLASSES = 20
# colour map
CLOTH_LABEL_COLORS = [(0, 0, 0)
                      # 0=Background
                , (128,0,0), (255,0,0), (0,85,0), (170,0,51), (255,85,0)
                      # 1=Hat,  2=Hair,    3=Glove, 4=Sunglasses, 5=UpperClothes
                , (0,0,85), (0,119,221), (85,85,0), (0,85,85), (85,51,0)
                      # 6=Dress, 7=Coat, 8=Socks, 9=Pants, 10=Jumpsuits
                , (52,86,128), (0,128,0), (0,0,255), (51,170,221), (0,255,255)
                      # 11=Scarf, 12=Skirt, 13=Face, 14=LeftArm, 15=RightArm
                , (85,255,170), (170,255,85), (255,255,0), (255,170,0)]
                # 16=LeftLeg, 17=RightLeg, 18=LeftShoe, 19=RightShoe

# take out sunglasses
CLOTH_LABEL_COLORS = CLOTH_LABEL_COLORS[:4] + CLOTH_LABEL_COLORS[5:]
N_CLOTH_CLASSES = 19


# image mean (FROM LIP_JPPNET, not our deepfashion. might be a little off)
IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)


def decode_cloth_labels(mask, num_images=-1, num_classes=N_CLOTH_CLASSES):
    """Decode batch of segmentation masks.
    AJ comment: Converts the tensor into a RGB image.

    Args:
      mask: result of inference after taking argmax.
      num_images: number of images to decode from the batch. Pass -1 for all (default: -1)

    Returns:
      A batch with num_images RGB images of the same size as the input.
    """
    n, c, h, w = mask.shape
    if num_images < 0:
        num_images = n
    assert n >= num_images, (
        "Batch size %d should be greater or equal than number of images to save %d."
        % (n, num_images)
    )
    outputs = np.zeros((num_images, h, w, 3), dtype=np.uint8)
    for i in range(num_images):
        img = Image.new("RGB", (w, h))
        pixels = img.load()
        # AJ: this enumerates the "rows" of the image (I think)
        for j_, j in enumerate(mask[i, 0, :, :]):
            for k_, k in enumerate(j):
                if k < N_CLOTH_CLASSES:
                    pixels[k_, j_] = CLOTH_LABEL_COLORS[k]
        outputs[i] = np.array(img)
    return outputs
