import os
import numpy as np
from PIL import Image
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split

# -------------------------------
# 1. Load and preprocess images
# -------------------------------
def load_images_from_folder(folder_path, image_size=(64, 64)):
    """Load grayscale images from a folder, resize to fixed dimensions."""
    images = []
    labels = []
    for label, person_folder in enumerate(os.listdir(folder_path)):
        person_path = os.path.join(folder_path, person_folder)
        if os.path.isdir(person_path):
            for filename in os.listdir(person_path):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    img_path = os.path.join(person_path, filename)
                    img = Image.open(img_path).convert('L')
                    img = img.resize(image_size, Image.Resampling.LANCZOS)
                    img_array = np.array(img).flatten()
                    images.append(img_array)
                    labels.append(label)
    return np.array(images).T, np.array(labels)  # each column is one image

def load_example_dataset():
    """Load a standard dataset for testing (Olivetti faces)."""
    data = fetch_olivetti_faces(shuffle=True, random_state=42)
    X = data.data.T  # each column: one face vector
    y = data.target
    return X, y

# -------------------------------
# 2. Core Eigenface algorithm
# -------------------------------
class EigenfaceRecognizer:
    def __init__(self, variance_retained=0.95):
        self.variance_retained = variance_retained
        self.mean_face = None
        self.eigenfaces = None
        self.eigenvalues = None
        self.k = None
        self.train_projections = None
        self.train_labels = None

    def fit(self, X, y):
        """
        X : ndarray of shape (d, N) where each column is a flattened face image
        y : ndarray of shape (N,) labels
        """
        N = X.shape[1]
        # Step 1: Compute mean face
        self.mean_face = np.mean(X, axis=1, keepdims=True)

        # Step 2: Center the data
        A = X - self.mean_face  # shape (d, N)

        # Step 3: Gram matrix trick (A^T A) instead of large covariance (A A^T)
        G = A.T @ A  # shape (N, N)

        # Step 4: Eigen decomposition of Gram matrix
        eigvals, eigvecs = np.linalg.eigh(G)

        # Sort eigenvalues and eigenvectors in descending order
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]

        # Step 5: Recover eigenfaces (eigenvectors of covariance)
        eigenfaces = A @ eigvecs
        # Normalize each eigenface to unit length
        for i in range(eigenfaces.shape[1]):
            norm = np.linalg.norm(eigenfaces[:, i])
            if norm > 1e-10:
                eigenfaces[:, i] /= norm

        # Step 6: Choose k based on variance retained
        total_variance = np.sum(eigvals)
        cumulative_variance = np.cumsum(eigvals) / total_variance
        self.k = np.searchsorted(cumulative_variance, self.variance_retained) + 1

        # Keep only first k eigenfaces
        self.eigenfaces = eigenfaces[:, :self.k]
        self.eigenvalues = eigvals[:self.k]

        # Step 7: Project training images into eigenface space
        self.train_projections = self._project(X)
        self.train_labels = y

    def _project(self, X):
        """Project images (centered) onto eigenfaces."""
        centered = X - self.mean_face
        return self.eigenfaces.T @ centered

    def predict(self, X, threshold=None):
        """
        Predict labels for new images.
        If threshold is given, return -1 for "unknown" if distance > threshold.
        """
        projections = self._project(X)
        predictions = []

        for p in projections.T:
            # Compute Euclidean distances to all training projections
            distances = np.linalg.norm(self.train_projections - p[:, np.newaxis], axis=0)
            min_idx = np.argmin(distances)
            min_dist = distances[min_idx]

            if threshold is not None and min_dist > threshold:
                predictions.append(-1)  # unknown
            else:
                predictions.append(self.train_labels[min_idx])
        return np.array(predictions)

    def reconstruct(self, X):
        """Reconstruct faces from eigenface coefficients."""
        projections = self._project(X)
        reconstructed = self.mean_face + self.eigenfaces @ projections
        return reconstructed

# -------------------------------
# 3. Evaluation utilities
# -------------------------------
def accuracy(y_true, y_pred):
    """Compute classification accuracy ignoring unknown (-1) if needed."""
    valid = y_pred != -1
    if np.sum(valid) == 0:
        return 0.0
    return np.mean(y_true[valid] == y_pred[valid])

def train_test_split_by_person(X, y, test_size=0.2, random_state=42):
    """Split data ensuring no person's images appear in both train and test."""
    unique_labels = np.unique(y)
    train_indices = []
    test_indices = []

    np.random.seed(random_state)
    for label in unique_labels:
        label_indices = np.where(y == label)[0]
        n_test = max(1, int(len(label_indices) * test_size))
        test_idx = np.random.choice(label_indices, n_test, replace=False)
        train_idx = np.setdiff1d(label_indices, test_idx)
        train_indices.extend(train_idx)
        test_indices.extend(test_idx)

    return X[:, train_indices], y[train_indices], X[:, test_indices], y[test_indices]

# -------------------------------
# 4. Main execution
# -------------------------------
if __name__ == "__main__":
    # Option A: Use example dataset (Olivetti faces)
    print("Loading Olivetti faces dataset...")
    X, y = load_example_dataset()   # X shape: (4096, 400), y shape: (400,)
    print(f"Dataset shape: {X.shape}, unique persons: {len(np.unique(y))}")

    # Split: 80% train, 20% test, same persons
    X_train, y_train, X_test, y_test = train_test_split_by_person(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Train: {X_train.shape[1]} samples, Test: {X_test.shape[1]} samples")

    # Train Eigenface recognizer
    recognizer = EigenfaceRecognizer(variance_retained=0.95)
    recognizer.fit(X_train, y_train)

    print(f"Optimal number of eigenfaces (k) = {recognizer.k}")

    # Predict on test set (no rejection)
    y_pred = recognizer.predict(X_test)
    acc = accuracy(y_test, y_pred)
    print(f"Recognition accuracy (no rejection): {acc:.4f}")

    # Optional: test with rejection (unknown faces)
    # Random noise as "unknown" faces
    unknown_faces = np.random.randn(X_train.shape[0], 10)
    threshold = 25  # tune based on training distances distribution
    y_pred_with_reject = recognizer.predict(unknown_faces, threshold=threshold)
    print(f"Unknown faces predictions ( -1 = rejected): {y_pred_with_reject}")

    # Reconstruction example
    sample_face = X_test[:, 0:1]
    reconstructed = recognizer.reconstruct(sample_face)
    print(f"Original shape: {sample_face.shape}, Reconstructed shape: {reconstructed.shape}")

    # Option B: Load your own dataset from folder
    # Uncomment and adjust path to use custom dataset
    """
    custom_folder = "path_to_your_folder"  # Each subfolder = one person
    image_size = (64, 64)
    X_custom, y_custom = load_images_from_folder(custom_folder, image_size)
    print(f"Custom dataset: {X_custom.shape}, unique people: {len(np.unique(y_custom))}")
    """
