import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import prnu
import numpy.fft as fft
from glob import glob

def resize_image(img, target_size=(2048, 2048)):
    """Resize an image to the target size using LANCZOS resampling."""
    return img.resize(target_size, Image.Resampling.LANCZOS)

def visualize_and_save_noise(noise, output_dir, image_name):
    """Visualize the noise pattern and its Fourier transform, then save to files."""
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Create figure for the noise pattern
    fig1 = plt.figure(figsize=(6, 6))
    plt.imshow(noise, cmap='gray')
    plt.colorbar()
    plt.title('Noise Pattern')
    noise_path = os.path.join(output_dir, f'{image_name}_noise.png')
    plt.savefig(noise_path)
    plt.close(fig1)

    # Perform Fourier transform and shift zero frequency component to the center
    f_noise = fft.fftshift(fft.fft2(noise))
    magnitude_spectrum = 20 * np.log(np.abs(f_noise) + 1)  # Adding 1 to avoid log(0)

    # Create figure for the Fourier transform magnitude spectrum
    fig2 = plt.figure(figsize=(6, 6))
    plt.imshow(magnitude_spectrum, cmap='hot')
    plt.colorbar()
    plt.title('Fourier Transform Magnitude Spectrum')
    spectrum_path = os.path.join(output_dir, f'{image_name}_spectrum.png')
    plt.savefig(spectrum_path)
    plt.close(fig2)

def extract_and_visualize_noise(img_path, output_dir):
    """ Extract and visualize PRNU noise from an image, then save the visualizations. """
    image_name = os.path.splitext(os.path.basename(img_path))[0]  # Extract base name without extension
    im = Image.open(img_path)
    im = resize_image(im)
    im_arr = np.asarray(im)
    if im_arr.dtype != np.uint8 or im_arr.ndim != 3:
        raise ValueError('Image must be RGB and of type uint8.')
    noise = prnu.extract_single(im_arr)
    visualize_and_save_noise(noise, output_dir, image_name)

def main():
    img_paths = glob(os.path.join('test/data/aiSanityCheck', '*.jpg'))
    for img_path in img_paths:
        extract_and_visualize_noise(img_path, 'test/data/SanityResult')

    img_paths = glob(os.path.join('test/data/sanityCheck', '*.jpg'))
    for img_path in img_paths:
        extract_and_visualize_noise(img_path, 'test/data/SanityResult')

if __name__ == '__main__':
    main()
