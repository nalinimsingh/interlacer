import numpy as np
import scipy.ndimage as ndimage


def add_rotation_and_translations(sl, coord_list, angle, num_pix):
    """Add k-space rotations and translations to input slice.

    At each line in coord_list in k-space, induce a rotation and translation.

    Args:
      sl(float): Numpy array of shape (n, n) containing input image data
      coord_list(int): Numpy array of (num_points) k-space line indices at which to induce motion
      angle(float): Numpy array of angles by which to rotate the input image; of shape (num_points)
      num_pix(float): List of horizontal and vertical translations by which to shift the input image; of shape (num_points, 2)

    Returns:
      sl_k_corrupt(float): Motion-corrupted k-space version of the input slice, of shape(n, n)

    """
    n = sl.shape[0]
    coord_list = np.concatenate([coord_list, [-1]])
    sl_k_true = np.fft.fftshift(np.fft.fft2(sl))

    sl_k_combined = np.zeros(sl.shape, dtype='complex64')
    sl_k_combined[:coord_list[0], :] = sl_k_true[:coord_list[0], :]

    for i in range(len(coord_list) - 1):
        sl_rotate = ndimage.rotate(sl, angle[i], reshape=False, mode='nearest')
        if(len(num_pix.shape) == 1):
            sl_moved = ndimage.interpolation.shift(
                sl_rotate, [0, num_pix[i]], mode='nearest')
        elif(num_pix.shape[1] == 2):
            sl_moved = ndimage.interpolation.shift(
                sl_rotate, [0, num_pix[i, 0]])
            sl_moved = ndimage.interpolation.shift(
                sl_moved, [num_pix[i, 1], 0])

        sl_k_after = np.fft.fftshift(np.fft.fft2(sl_moved))
        if(coord_list[i + 1] != -1):
            sl_k_combined[coord_list[i]:coord_list[i + 1],
                          :] = sl_k_after[coord_list[i]:coord_list[i + 1], :]
            if(coord_list[i] <= int(n / 2) and int(n / 2) < coord_list[i + 1]):
                sl_k_true = sl_k_after
        else:
            sl_k_combined[coord_list[i]:, :] = sl_k_after[coord_list[i]:, :]
            if(coord_list[i] <= int(n / 2)):
                sl_k_true = sl_k_after

    return sl_k_combined, sl_k_true
