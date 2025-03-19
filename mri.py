import os
import numpy as np
from skimage import io, transform
from sklearn.model_selection import train_test_split

# Define paths
dataset_dir = "C:/JAYASOORYA/RESEARCH SOURCE/brain_tumor_dataset"
output_dir = "C:/JAYASOORYA/RESEARCH SOURCE"
os.makedirs(output_dir, exist_ok=True)

# Load and preprocess images
images = []
labels = []

for label, class_name in enumerate(['yes', 'no']):  # Assuming 'yes' = tumor, 'no' = no tumor
    class_dir = os.path.join(dataset_dir, class_name)
    for image_name in os.listdir(class_dir):
        image_path = os.path.join(class_dir, image_name)
        image = io.imread(image_path, as_gray=True)  # Read as grayscale
        image = transform.resize(image, (128, 128))  # Resize to 128x128
        images.append(image)
        labels.append(label)

# Convert to numpy arrays
images = np.array(images)
labels = np.array(labels)

# Save preprocessed data
np.save(os.path.join(output_dir, 'mri_images.npy'), images)
np.save(os.path.join(output_dir, 'mri_labels.npy'), labels)

print(f"Preprocessed data saved to {output_dir}")