import torch
import os


def save_models(dir, epoch, batches_done=None, **kwargs):
    """
    Save a model to a dir
    :param dir: output dir
    :param epoch: epoch number
    :param batches_done: num batches finished (optional)
    :param kwargs: model name to model
    :return:
    """
    os.makedirs(dir, exist_ok=True)
    # Save model checkpoints
    for name, model in kwargs:
        fname = f"{name}_{epoch:02d}"
        if batches_done:
            fname += f"_{batches_done:05d}"
        fname += ".pth"
        torch.save(model.state_dict(), os.path.join(dir, fname))
