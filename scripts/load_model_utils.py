import os

import numpy as np
import pandas as pd

from tensorboard.backend.event_processing import event_accumulator


def get_val_event_log(base_dir):
    """

    Args:
      base_dir: Path to model training directory

    Returns:
      (str) Path to validation tensorboard data

    """
    tb_str = os.path.join(base_dir,'tensorboard/validation/')
    return tb_str


def get_best_ckpt(model_path):
    """

    Args:
      model_path: Path to model training directory

    Returns:
      (int) epoch with lowest validation loss

    """
    x = event_accumulator.EventAccumulator(
        path=get_val_event_log(model_path), size_guidance={
            event_accumulator.SCALARS: 0}, purge_orphaned_data=False)
    x.Reload()

    img_l1 = pd.DataFrame(x.Scalars('epoch_loss')).value[5::5]
        
    best_epoch = 5 * (np.argmin(img_l1) + 1) # epochs stored in intervals of 5
    return best_epoch