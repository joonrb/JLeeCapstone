import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import numpy.fft as fft
from glob import glob
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def resize_image(img, target_size=(1024, 1024)):
    """Resize an image to the target size using LANCZOS resampling."""
    return img.resize(target_size, Image.Resampling.LANCZOS)

def extract_features(im_arr):
    """Extract Fourier transform features from an image array."""
    # Applying FFT and shifting the zero frequency to the center
    f_transform = fft.fftshift(fft.fft2(im_arr))
    magnitude_spectrum = np.log(np.abs(f_transform) + 1)

    # Extract the top right quarter of the Fourier transform
    mid_x, mid_y = magnitude_spectrum.shape[0] // 2, magnitude_spectrum.shape[1] // 2
    top_right_quarter = magnitude_spectrum[:mid_x, mid_y:]

    return top_right_quarter.flatten()

def process_images(img_paths):
    """Process images and extract Fourier features."""
    features = []
    for img_path in img_paths:
        im = Image.open(img_path)
        im = resize_image(im)
        im_arr = np.asarray(im)
        if im_arr.dtype != np.uint8 or im_arr.ndim != 3:
            raise ValueError('Image must be RGB and of type uint8.')
        feature = extract_features(im_arr)
        features.append(feature)
    return features

def cluster_and_visualize(features):
    """Cluster features and visualize with PCA."""
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    kmeans = KMeans(n_clusters=2, random_state=42)
    labels = kmeans.fit_predict(features_scaled)

    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(features_scaled)

    plt.figure(figsize=(8, 6))
    plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=labels, cmap='viridis', marker='o', edgecolor='k')
    plt.title('PCA Visualization of Clustering Results')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.colorbar()
    plt.show()
    
    return labels

def print_labels(img_paths, labels):
    """Print the cluster labels for each image path."""
    for img_path, label in zip(img_paths, labels):
        print(f'{img_path}: Cluster {label}')

def main():
    ai_img_paths = glob('test/data/ai-jpg/*.jpg')
    human_img_paths = glob('test/data/human-jpg/*.jpg')

    ai_features = process_images(ai_img_paths)
    human_features = process_images(human_img_paths)

    all_img_paths = ai_img_paths + human_img_paths
    all_features = ai_features + human_features
    labels = cluster_and_visualize(all_features)
    print_labels(all_img_paths, labels)

if __name__ == '__main__':
    main()
