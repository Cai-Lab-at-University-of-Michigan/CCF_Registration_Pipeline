import numpy as np
import pywt
from scipy import ndimage, fftpack
from skimage.filters.thresholding import threshold_otsu


def fft(data, axis=-1, shift=True):
    """Computes the 1D Fast Fourier Transform of an input array
    Parameters
    ----------
    data : ndarray
        input array to transform
    axis : int (optional)
        axis to perform the 1D FFT over
    shift : bool
        indicator for centering the DC component
    Returns
    -------
    fdata : ndarray
        transformed data
    """
    fdata = fftpack.rfft(data, axis=axis)
    # fdata = fftpack.rfft(fdata, axis=0)
    if shift:
        fdata = fftpack.fftshift(fdata)
    return fdata


def ifft(fdata, axis=-1):
    # fdata = fftpack.irfft(fdata, axis=0)
    return fftpack.irfft(fdata, axis=axis)


def fft2(data, shift=True):
    """Computes the 2D Fast Fourier Transform of an input array
    Parameters
    ----------
    data : ndarray
        data to transform
    shift : bool
        indicator for center the DC component
    Returns
    -------
    fdata : ndarray
        transformed data
    """
    fdata = fftpack.fft2(data)
    if shift:
        fdata = fftpack.fftshift(fdata)
    return fdata


def ifft2(fdata):
    return fftpack.ifft2(fdata)


def magnitude(fdata):
    return np.sqrt(np.real(fdata) ** 2 + np.imag(fdata) ** 2)


def notch(n, sigma):
    """Generates a 1D gaussian notch filter `n` pixels long
    Parameters
    ----------
    n : int
        length of the gaussian notch filter
    sigma : float
        notch width
    Returns
    -------
    g : ndarray
        (n,) array containing the gaussian notch filter
    """
    if n <= 0:
        raise ValueError('n must be positive')
    else:
        n = int(n)
    if sigma <= 0:
        raise ValueError('sigma must be positive')
    x = np.arange(n)
    g = 1 - np.exp(-x ** 2 / (2 * sigma ** 2))
    return g


def gaussian_filter(shape, sigma):
    """Create a gaussian notch filter
    Parameters
    ----------
    shape : tuple
        shape of the output filter
    sigma : float
        filter bandwidth
    Returns
    -------
    g : ndarray
        the impulse response of the gaussian notch filter
    """
    g = notch(n=shape[-1], sigma=sigma)
    g_mask = np.broadcast_to(g, shape).copy()
    return g_mask


def wavedec(img, wavelet, level=None):
    """Decompose `img` using discrete (decimated) wavelet transform using `wavelet`
    Parameters
    ----------
    img : ndarray
        image to be decomposed into wavelet coefficients
    wavelet : str
        name of the mother wavelet
    level : int (optional)
        number of wavelet levels to use. Default is the maximum possible decimation
    Returns
    -------
    coeffs : list
        the approximation coefficients followed by detail coefficient tuple for each level
    """
    return pywt.wavedec2(img, wavelet, mode='symmetric', level=level, axes=(-2, -1))


def waverec(coeffs, wavelet):
    """Reconstruct an image using a multilevel 2D inverse discrete wavelet transform
    Parameters
    ----------
    coeffs : list
        the approximation coefficients followed by detail coefficient tuple for each level
    wavelet : str
        name of the mother wavelet
    Returns
    -------
    img : ndarray
        reconstructed image
    """
    return pywt.waverec2(coeffs, wavelet, mode='symmetric', axes=(-2, -1))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def foreground_fraction(img, center, crossover, smoothing):
    z = (img - center) / crossover
    f = sigmoid(z)
    return ndimage.gaussian_filter(f, sigma=smoothing)


def filter_subband(img, sigma, level, wavelet):
    def inner_fft(c, frac):
        s = c.shape[0] * frac
        fc = fft(c, shift=False)
        g = gaussian_filter(shape=fc.shape, sigma=s)
        fc_filt = fc * g
        c_filt = ifft(fc_filt)
        return c_filt

    img_log = np.log(1 + img)

    if level == 0:
        coeffs = wavedec(img_log, wavelet)
    else:
        coeffs = wavedec(img_log, wavelet, level)
    approx = coeffs[0]
    detail = coeffs[1:]

    coeffs_filt = [approx]
    for ch, cv, cd in detail:
        ch_filt = inner_fft(ch, sigma / img.shape[0])
        cv_filt = inner_fft(cv, sigma / img.shape[1])
        cd_filt = inner_fft(cd, sigma / max(img.shape))
        coeffs_filt.append((ch_filt, cv_filt, cd_filt))

    img_log_filtered = waverec(coeffs_filt, wavelet)
    return np.exp(img_log_filtered) - 1


def apply_flat(img, flat):
    return (img / flat).astype(img.dtype)


def filter_stripes(img, sigma=[256, 512], level=6, wavelet='db4', crossover=10, threshold=-1, flat=None, dark=0,
                   smoothing=1):
    """Filter horizontal streaks using wavelet-FFT filter
    Parameters
    ----------
    img : ndarray
        input image array to filter
    sigma : float or list
        filter bandwidth(s) in pixels (larger gives more filtering)
    level : int
        number of wavelet levels to use
    wavelet : str
        name of the mother wavelet
    crossover : float
        intensity range to switch between filtered background and unfiltered foreground
    threshold : float
        intensity value to separate background from foreground. Default is Otsu
    flat : ndarray
        reference image for illumination correction. Must be same shape as input images. Default is None
    dark : float
        Intensity to subtract from the images for dark offset. Default is 0.
    smoothing: float
        Smoothing factor to generate Gaussian filter
    Returns
    -------
    fimg : ndarray
        filtered image
    """

    data_type = img.dtype

    if threshold == -1:
        try:
            threshold = threshold_otsu(img)
        except:
            threshold = 1

    img = np.array(img, dtype=np.float)

    pady, padx = [_ % 2 for _ in img.shape]
    if pady == 1 or padx == 1:
        img = np.pad(img, ((0, pady), (0, padx)), mode="edge")

    sigma1 = sigma[0]  # foreground
    sigma2 = sigma[1]  # background
    if sigma1 > 0:
        if sigma2 > 0:
            if sigma1 == sigma2:  # Single band
                fimg = filter_subband(img, sigma1, level, wavelet)
            else:  # Dual-band
                background = np.clip(img, None, threshold)
                foreground = np.clip(img, threshold, None)
                background_filtered = filter_subband(background, sigma[1], level, wavelet)
                foreground_filtered = filter_subband(foreground, sigma[0], level, wavelet)
                # Smoothed homotopy
                f = foreground_fraction(img, threshold, crossover, smoothing=smoothing)
                fimg = foreground_filtered * f + background_filtered * (1 - f)
        else:  # Foreground filter only
            foreground = np.clip(img, threshold, None)
            foreground_filtered = filter_subband(foreground, sigma[0], level, wavelet)
            # Smoothed homotopy
            f = foreground_fraction(img, threshold, crossover, smoothing=smoothing)
            fimg = foreground_filtered * f + img * (1 - f)
    else:
        if sigma2 > 0:  # Background filter only
            background = np.clip(img, None, threshold)
            background_filtered = filter_subband(background, sigma[1], level, wavelet)
            # Smoothed homotopy
            f = foreground_fraction(img, threshold, crossover, smoothing=smoothing)
            fimg = img * f + background_filtered * (1 - f)
        else:
            # sigma1 and sigma2 are both 0, so skip the destriping
            fimg = img

    # Subtract the dark offset fiirst
    if dark > 0:
        fimg = fimg - dark

    # Divide by the flat
    if flat is not None:
        fimg = apply_flat(fimg, flat)

    # Convert to same bit image as original image
    if data_type == np.uint8:
        np.clip(fimg, 0, 255, out=fimg)  # Clip to 6-bit unsigned range
    elif data_type == np.uint16:
        np.clip(fimg, 0, 2 ** 16 - 1, out=fimg)  # Clip to 6-bit unsigned range
    else:
        raise RuntimeError(f'image data type should be 8bit or 16bit, currently is {data_type}')

    fimg = fimg.astype(data_type)

    if padx > 0:
        fimg = fimg[:, :-padx]
    if pady > 0:
        fimg = fimg[:-pady]

    return fimg
