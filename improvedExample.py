import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import numpy.fft as fft
from glob import glob
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import scipy.stats  # Correctly importing scipy.stats for statistical functions
import gc  # Import garbage collector

def resize_image(img, target_size=(256, 256)):  # Reduced image size for better memory management
    """Resize an image to the target size using LANCZOS resampling."""
    return img.resize(target_size, Image.Resampling.LANCZOS)

def extract_features(im_arr):
    """Extract enhanced Fourier transform features from an image array."""
    f_transform = fft.fftshift(fft.fft2(im_arr))
    magnitude_spectrum = np.log(np.abs(f_transform) + 1)
    mid_x, mid_y = magnitude_spectrum.shape[0] // 2, magnitude_spectrum.shape[1] // 2
    top_right_quarter = magnitude_spectrum[:mid_x, mid_y:]

    # Calculate additional statistical features from the flattened top right quarter
    flattened_spectrum = top_right_quarter.flatten()
    features = {
        'mean': np.mean(flattened_spectrum),
        'std': np.std(flattened_spectrum),
        'skewness': scipy.stats.skew(flattened_spectrum),
        'kurtosis': scipy.stats.kurtosis(flattened_spectrum),
        'entropy': scipy.stats.entropy(flattened_spectrum + np.finfo(float).eps)  # Add a small epsilon to avoid log(0)
    }

    # Combine the flattened spectrum with additional features
    enhanced_features = np.append(flattened_spectrum, [
        features['mean'], features['std'], features['skewness'], features['kurtosis'], features['entropy']
    ])
    return enhanced_features

def process_images(img_paths):
    """Process images and extract enhanced Fourier features."""
    features = []
    for img_path in img_paths:
        im = Image.open(img_path)
        im = resize_image(im)
        im_arr = np.asarray(im)
        if im_arr.dtype != np.uint8 or im_arr.ndim != 3:
            raise ValueError('Image must be RGB and of type uint8.')
        feature = extract_features(im_arr)
        features.append(feature)
        del im, im_arr  # Explicitly manage memory
        gc.collect()  # Invoke garbage collector
    return features

def cluster_and_visualize(features):
    """Use Gaussian Mixture Model for clustering and PCA for visualization."""
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    gmm = GaussianMixture(n_components=2, n_init=1, random_state=42)
    labels = gmm.fit_predict(features_scaled)

    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(features_scaled)

    plt.figure(figsize=(8, 6))
    plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=labels, cmap='viridis', marker='o', edgecolor='k')
    plt.title('PCA Visualization of GMM Clustering Results')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.colorbar()
    plt.show()
    gc.collect()  # Clean up memory after heavy operations
    return labels

def print_labels(img_paths, labels):
    """Print the cluster labels for each image path."""
    for img_path, label in zip(img_paths, labels):
        print(f'{img_path}: Cluster {label}')

def main():
    ai_img_paths = glob('test/data/ai-jpg/*.jpg')
    human_img_paths = glob('test/data/human-jpg/*.jpg')

    all_img_paths = ai_img_paths + human_img_paths
    all_features = process_images(all_img_paths)
    labels = cluster_and_visualize(all_features)
    print_labels(all_img_paths, labels)

if __name__ == '__main__':
    main()
