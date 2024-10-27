import numpy as np
from glob import glob
from PIL import Image
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
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

def run_hierarchical_clustering(features, n_clusters=2):
    """Run hierarchical clustering on the features."""
    scaler = StandardScaler()  # Normalize the data
    features_scaled = scaler.fit_transform(features)
    
    # Fit hierarchical clustering model
    cluster = AgglomerativeClustering(n_clusters=n_clusters)
    labels = cluster.fit_predict(features_scaled)
    
    return labels

def visualize_with_pca(features, labels=None):
    """ Standardize the data (important for PCA) and visualize using PCA. """
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Apply PCA to reduce the dimensions to 2D
    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(features_scaled)
    
    # Plot the 2D projection
    plt.figure(figsize=(8, 6))
    
    if labels is not None:
        scatter = plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=labels, cmap='viridis', marker='o', s=50, edgecolor='k', label='Clusters')
        plt.title('PCA Visualization with Clustering Labels')
        plt.legend(handles=scatter.legend_elements()[0], labels=set(labels), title="Clusters")
    else:
        plt.scatter(reduced_features[:, 0], reduced_features[:, 1], marker='o', s=50, edgecolor='k')
        plt.title('PCA Visualization of PRNU Features')

    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.colorbar(scatter)
    plt.show()

def main():
    ai_features = load_and_extract_features('test/data/ai-jpg')
    human_features = load_and_extract_features('test/data/human-jpg')
    
    if ai_features.size > 0 and human_features.size > 0:
        all_features = np.vstack((ai_features, human_features))
        labels = run_hierarchical_clustering(all_features) 
        print("Clustering Labels:", labels)
        all_features = np.vstack((ai_features, human_features))
        visualize_with_pca(all_features, labels)
    else:
        print("Failed to extract sufficient features for clustering.")

if __name__ == '__main__':
    main()
