import numpy as np
from glob import glob
from PIL import Image
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import prnu

def resize_image(img, target_size=(1024, 1024)):
    """Resize an image to the target size using LANCZOS resampling."""
    return img.resize(target_size, Image.Resampling.LANCZOS)

def extract_prnu_features(img_path):
    """ Extract PRNU features from an image as a flattened array. """
    im = Image.open(img_path)
    im_arr = np.asarray(im)
    if im_arr.dtype != np.uint8 or im_arr.ndim != 3:
        raise ValueError('Image must be RGB and of type uint8.')
    noise = prnu.extract_single(im_arr)
    return noise.flatten()

"""
def load_and_extract_features(directory):
    #Load images and extract features from a directory.
    img_paths = glob(f'{directory}/*.jpg')
    features = []
    for path in img_paths:
        try:
            feature = extract_prnu_features(path)
            features.append(feature)
        except Exception as e:
            print(f"Failed to process {path}: {e}")
    return np.array(features)
"""

def load_and_extract_features(directory):
    """ Load images, resize them to be consistent, and extract PRNU features. """
    img_paths = glob(f'{directory}/*.jpg')
    imgs = []
    for path in img_paths:
        try:
            im = Image.open(path)
            im = resize_image(im, target_size=(1024, 1024))
            imgs.append(np.asarray(im))
        except Exception as e:
            print(f"Failed to process {path}: {e}")
    
    features = prnu.extract_multiple_aligned(imgs)
    return features

    
def run_gmm_clustering(features, n_clusters=2):
    """Run Gaussian Mixture Model (GMM) clustering on the features."""
    scaler = StandardScaler()  # Normalize the data as GMM is sensitive to scaling
    features_scaled = scaler.fit_transform(features)
    
    # Fit Gaussian Mixture Model
    gmm = GaussianMixture(n_components=n_clusters, random_state=42)
    gmm.fit(features_scaled)
    
    # Predict the cluster labels
    labels = gmm.predict(features_scaled)
    
    return labels

def visualize_with_pca(features, labels=None):
    # Standardize the data (important for PCA)
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Apply PCA to reduce the dimensions to 2D
    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(features_scaled)
    
    # Plot the 2D projection
    plt.figure(figsize=(8, 6))
    
    if labels is not None:
        plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=labels, cmap='viridis', marker='o', s=50, edgecolor='k')
        plt.title('PCA Visualization with Clustering Labels')
    else:
        plt.scatter(reduced_features[:, 0], reduced_features[:, 1], marker='o', s=50, edgecolor='k')
        plt.title('PCA Visualization of PRNU Features')

    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.colorbar()
    plt.show()


def main():
    ai_features = load_and_extract_features('test/data/ai-jpg')
    human_features = load_and_extract_features('test/data/human-jpg')
    
    if ai_features.size > 0 and human_features.size > 0:
        all_features = np.vstack((ai_features, human_features))
        labels = run_gmm_clustering(all_features)  # Use GMM clustering
        print("Clustering Labels:", labels)
        all_features = np.vstack((ai_features, human_features))
        visualize_with_pca(all_features, labels)
    else:
        print("Failed to extract sufficient features for clustering.")

if __name__ == '__main__':
    main()
