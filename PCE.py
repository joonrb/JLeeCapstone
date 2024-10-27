import numpy as np
from glob import glob
from PIL import Image
import prnu

def resize_image(img, target_size=(1024, 1024)):
    """Resize an image to the target size using LANCZOS resampling."""
    return img.resize(target_size, Image.Resampling.LANCZOS)

def extract_prnu_features(img_path):
    """ Extract PRNU features from an image as a 2D array. """
    im = Image.open(img_path)
    im = resize_image(im)
    im_arr = np.asarray(im)
    if im_arr.dtype != np.uint8 or im_arr.ndim != 3:
        raise ValueError('Image must be RGB and of type uint8.')
    noise = prnu.extract_single(im_arr)
    if noise.ndim != 2:
        raise ValueError('Extracted noise must be 2-dimensional')
    return noise

def load_and_extract_features(directory):
    """ Load images and extract PRNU features. """
    img_paths = glob(f'{directory}/*.jpg')
    features = []
    for path in img_paths:
        try:
            feature = extract_prnu_features(path)
            features.append(feature)
        except Exception as e:
            print(f"Failed to process {path}: {e}")
    return np.array(features)

def calculate_cross_correlation_and_pce(reference, features):
    """ Calculate cross-correlation and PCE between reference PRNU and each feature set, and classify. """
    PCE_THRESHOLD = 10  # Set this based on experimental data or a validation study
    results = []
    for feature in features:
        if reference.ndim != 2 or feature.ndim != 2:
            raise ValueError("Both reference and feature must be 2-dimensional")
        cc = prnu.crosscorr_2d(reference, feature)
        pce = prnu.pce(cc)
        classification = 'Camera' if pce['pce'] >= PCE_THRESHOLD else 'AI'
        results.append((cc, pce['pce'], classification))
    return results

def main():
    # Assuming there's a reference fingerprint available
    reference_features = load_and_extract_features('test/data/ai-reference')
    reference_feature = np.mean(reference_features, axis=0) if reference_features.ndim > 2 else reference_features

    ai_features = load_and_extract_features('test/data/ai-jpg')
    camera_features = load_and_extract_features('test/data/human-jpg')

    ai_results = calculate_cross_correlation_and_pce(reference_feature, ai_features)
    camera_results = calculate_cross_correlation_and_pce(reference_feature, camera_features)

    # Output results
    print("AI Images Results:")
    for result in ai_results:
        print(f"Classified as: {result[2]}")

    print("Camera Images Results:")
    for result in camera_results:
        print(f"Classified as: {result[2]}")

if __name__ == '__main__':
    main()
