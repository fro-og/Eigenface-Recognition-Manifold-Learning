import os
import numpy as np
import random
from PIL import Image

# --------------------------------------------------
# 1. Load YOUR dataset from dataset/s1/, dataset/s2/, etc.
# --------------------------------------------------
def load_my_dataset(dataset_path="dataset", images_per_person=10):
    """
    Load your dataset from subfolders.
    Expected structure:
        dataset/
            s1/
                1.pgm (or .jpg, .png)
                2.pgm
                ...
                10.pgm
            s2/
                1.pgm
                ...
    
    Returns:
        X: ndarray of shape (d, N) where each column is a flattened face
        y: ndarray of shape (N,) with integer labels (0, 1, 2, ...)
        person_names: list of folder names
        img_shape: tuple (height, width) of original images
    """
    images = []
    labels = []
    person_names = []
    img_shape = None
    
    # Get all person folders (s1, s2, ...)
    folders = sorted([f for f in os.listdir(dataset_path) 
                     if os.path.isdir(os.path.join(dataset_path, f))])
    
    print(f"Found {len(folders)} persons in dataset")
    
    for label, folder in enumerate(folders):
        folder_path = os.path.join(dataset_path, folder)
        person_names.append(folder)
        
        # Load images for this person (expecting 1.pgm, 2.pgm, ..., 10.pgm)
        person_images = []
        for i in range(1, images_per_person + 1):
            # Try different extensions
            img_path = None
            for ext in ['.pgm', '.png', '.jpg', '.jpeg', '.bmp']:
                test_path = os.path.join(folder_path, f"{i}{ext}")
                if os.path.exists(test_path):
                    img_path = test_path
                    break
            
            if img_path is None:
                print(f"Warning: Missing image {i} for {folder}")
                continue
                
            img = Image.open(img_path).convert('L')  # grayscale
            if img_shape is None:
                img_shape = img.size[::-1]  # (height, width)
            img_array = np.array(img).flatten()
            person_images.append(img_array)
        
        if len(person_images) == images_per_person:
            images.extend(person_images)
            labels.extend([label] * images_per_person)
        else:
            print(f"Warning: {folder} has only {len(person_images)} images, expected {images_per_person}")
    
    X = np.array(images).T  # each column = one face vector
    y = np.array(labels)
    
    print(f"\nLoaded {X.shape[1]} total images")
    print(f"Image dimension: {X.shape[0]} pixels")
    print(f"Image shape: {img_shape[0]}x{img_shape[1]}")
    print(f"Number of persons: {len(np.unique(y))}")
    
    return X, y, person_names, img_shape

# --------------------------------------------------
# 2. Eigenface recognizer (pure linear algebra)
# --------------------------------------------------
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
        X: (d, N) each column = flattened face
        y: (N,) labels
        """
        N = X.shape[1]
        
        print(f"\nTraining with {N} images...")
        
        # Step 1: Mean face
        self.mean_face = np.mean(X, axis=1, keepdims=True)
        
        # Step 2: Center data
        A = X - self.mean_face
        
        # Step 3: Gram matrix trick (A^T A) instead of huge (A A^T)
        G = A.T @ A
        
        # Step 4: Eigendecomposition of Gram matrix
        eigvals, eigvecs = np.linalg.eigh(G)
        
        # Sort descending
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        
        # Step 5: Recover eigenfaces (eigenvectors of covariance)
        eigenfaces = A @ eigvecs
        
        # Normalize each eigenface
        for i in range(eigenfaces.shape[1]):
            norm = np.linalg.norm(eigenfaces[:, i])
            if norm > 1e-10:
                eigenfaces[:, i] /= norm
        
        # Step 6: Choose k to retain variance_retained
        total_var = np.sum(eigvals)
        cum_var = np.cumsum(eigvals) / total_var
        self.k = np.searchsorted(cum_var, self.variance_retained) + 1
        
        # Keep only first k eigenfaces
        self.eigenfaces = eigenfaces[:, :self.k]
        self.eigenvalues = eigvals[:self.k]
        
        # Step 7: Project training images
        self.train_projections = self._project(X)
        self.train_labels = y
        
        print(f"Kept {self.k} eigenfaces (retained {self.variance_retained*100:.0f}% variance)")

    def _project(self, X):
        """Project centered images onto eigenfaces."""
        centered = X - self.mean_face
        return self.eigenfaces.T @ centered

    def predict(self, X):
        """
        Predict labels for test images.
        Returns predicted labels and distances.
        """
        proj = self._project(X)
        predictions = []
        distances = []
        
        for i in range(proj.shape[1]):
            p = proj[:, i:i+1]
            # Euclidean distances to all training projections
            dists = np.linalg.norm(self.train_projections - p, axis=0)
            min_idx = np.argmin(dists)
            min_dist = dists[min_idx]
            distances.append(min_dist)
            predictions.append(self.train_labels[min_idx])
        
        return np.array(predictions), np.array(distances)
    
    def reconstruct(self, X):
        """Reconstruct faces from eigenface coefficients."""
        proj = self._project(X)
        return self.mean_face + self.eigenfaces @ proj

# --------------------------------------------------
# 3. Main recognition experiment with YOUR dataset
# --------------------------------------------------
def main():
    print("="*60)
    print("EIGENFACE RECOGNITION WITH YOUR DATASET")
    print("="*60)
    
    # Step 1: Load YOUR dataset
    print("\n[1] Loading your dataset from 'dataset/' folder...")
    print("-"*60)
    
    X_all, y_all, person_names, img_shape = load_my_dataset(dataset_path="dataset", images_per_person=10)
    
    if X_all.shape[1] == 0:
        print("Error: No images found! Please check your dataset folder structure.")
        print("Expected structure: dataset/s1/1.pgm, dataset/s1/2.pgm, ..., dataset/s2/1.pgm, ...")
        return
    
    num_persons = len(person_names)
    images_per_person = 10
    height, width = img_shape
    
    print(f"\nDataset summary:")
    print(f"  - {num_persons} persons")
    print(f"  - {images_per_person} images per person")
    print(f"  - Total images: {X_all.shape[1]}")
    print(f"  - Image size: {width}x{height} pixels")
    
    # Step 2: Prepare training data (first 9 images of each person)
    print("\n[2] Preparing training data (first 9 images of each person)...")
    print("-"*60)
    
    X_train = []
    y_train = []
    
    for person_id in range(num_persons):
        # Take first 9 images for training (indices 0-8)
        start_idx = person_id * images_per_person
        train_indices = list(range(start_idx, start_idx + 9))
        
        for idx in train_indices:
            X_train.append(X_all[:, idx:idx+1])
            y_train.append(y_all[idx])
    
    X_train = np.hstack(X_train)  # stack columns
    y_train = np.array(y_train)
    
    print(f"Training set: {X_train.shape[1]} images ({9} per person)")
    
    # Step 3: Randomly select a person for testing (using their 10th image)
    print("\n[3] Selecting random person for testing...")
    print("-"*60)
    
    random.seed(42)  # You can change or remove this for true randomness
    test_person_id = random.randint(0, num_persons - 1)
    
    # Get the 10th image of this person (index 9)
    test_idx = test_person_id * images_per_person + 9
    X_test = X_all[:, test_idx:test_idx+1]
    y_test_true = y_all[test_idx]
    
    person_name = person_names[test_person_id]
    print(f"\nSelected test person: {person_name} (ID: {test_person_id})")
    print(f"Using their 10th image for testing")
    print(f"True label: {y_test_true}")
    
    # Step 4: Train Eigenface recognizer
    print("\n[4] Training Eigenface recognizer...")
    print("-"*60)
    
    recognizer = EigenfaceRecognizer(variance_retained=0.95)
    recognizer.fit(X_train, y_train)
    
    # Step 5: Test recognition
    print("\n[5] Testing recognition...")
    print("-"*60)
    
    y_pred, distances = recognizer.predict(X_test)
    
    print(f"\n{'='*60}")
    print("RECOGNITION RESULT")
    print(f"{'='*60}")
    print(f"\nTest image belongs to: {person_name} (Person ID: {test_person_id})")
    print(f"Predicted as: {person_names[y_pred[0]]} (Person ID: {y_pred[0]})")
    
    if y_pred[0] == y_test_true:
        print(f"\n✓ SUCCESS! The 10th image was correctly recognized!")
    else:
        print(f"\n✗ FAILURE! The 10th image was misclassified.")
    
    print(f"\nDistance to closest training image: {distances[0]:.4f}")
    
    # Step 6: Show distances to all persons
    print("\n[6] Distance analysis...")
    print("-"*60)
    
    print("\nRanking of closest persons:")
    person_distances = []
    
    for person_id in range(num_persons):
        # Get all training projections for this person
        person_mask = (y_train == person_id)
        if np.sum(person_mask) > 0:  # Make sure there are training images for this person
            person_projections = recognizer.train_projections[:, person_mask]
            
            # Average projection for this person
            avg_projection = np.mean(person_projections, axis=1, keepdims=True)
            
            # Distance from test image to this person's average
            test_proj = recognizer._project(X_test)
            dist = np.linalg.norm(test_proj - avg_projection)
            person_distances.append((person_id, dist, person_names[person_id]))
    
    # Sort by distance
    person_distances.sort(key=lambda x: x[1])
    
    for rank, (pid, dist, name) in enumerate(person_distances[:5], 1):
        match_marker = "✓" if pid == test_person_id else " "
        print(f"  {rank}. {match_marker} {name}: distance = {dist:.4f}")
    
    # Step 7: Optional - Show visualization
    print("\n[7] Generating visualizations...")
    print("-"*60)
    
    try:
        import matplotlib.pyplot as plt
        
        # Create figure with proper sizing
        fig = plt.figure(figsize=(14, 10))
        
        # 1. Mean face
        ax1 = fig.add_subplot(2, 3, 1)
        mean_face_img = recognizer.mean_face.reshape(height, width)
        ax1.imshow(mean_face_img, cmap='gray')
        ax1.set_title("Mean Face", fontsize=12)
        ax1.axis('off')
        
        # 2. Test image (10th)
        ax2 = fig.add_subplot(2, 3, 2)
        test_img = X_test.reshape(height, width)
        ax2.imshow(test_img, cmap='gray')
        ax2.set_title(f"Test Image (Person {person_name} - 10th)", fontsize=12)
        ax2.axis('off')
        
        # 3. Closest training match
        ax3 = fig.add_subplot(2, 3, 3)
        closest_idx = np.argmin(distances)
        closest_img = X_train[:, closest_idx:closest_idx+1].reshape(height, width)
        ax3.imshow(closest_img, cmap='gray')
        ax3.set_title(f"Closest Match ({person_names[y_train[closest_idx]]})", fontsize=12)
        ax3.axis('off')
        
        # 4. Reconstructed face
        ax4 = fig.add_subplot(2, 3, 4)
        reconstructed = recognizer.reconstruct(X_test)
        reconstructed_img = reconstructed.reshape(height, width)
        ax4.imshow(reconstructed_img, cmap='gray')
        ax4.set_title(f"Reconstructed (k={recognizer.k})", fontsize=12)
        ax4.axis('off')
        
        # 5. Difference (original - reconstructed)
        ax5 = fig.add_subplot(2, 3, 5)
        diff_img = np.abs(test_img - reconstructed_img)
        ax5.imshow(diff_img, cmap='hot')
        ax5.set_title("Reconstruction Error", fontsize=12)
        ax5.axis('off')
        
        # 6. Cumulative variance plot
        ax6 = fig.add_subplot(2, 3, 6)
        total_var = np.sum(recognizer.eigenvalues)
        cum_var = np.cumsum(recognizer.eigenvalues) / total_var
        ax6.plot(range(1, len(cum_var) + 1), cum_var, 'b-', linewidth=2)
        ax6.axhline(y=0.95, color='r', linestyle='--', label='95% variance')
        ax6.axvline(x=recognizer.k, color='g', linestyle='--', label=f'k={recognizer.k}')
        ax6.set_xlabel('Number of Eigenfaces', fontsize=10)
        ax6.set_ylabel('Cumulative Variance', fontsize=10)
        ax6.set_title('Variance Retained', fontsize=12)
        ax6.grid(True, alpha=0.3)
        ax6.legend()
        
        plt.suptitle(f"Eigenface Recognition - Result: {'✓ CORRECT' if y_pred[0] == y_test_true else '✗ WRONG'}", 
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        # Show top 9 eigenfaces in a separate figure
        fig2, axes2 = plt.subplots(3, 3, figsize=(10, 10))
        for i, ax in enumerate(axes2.flat):
            if i < min(9, recognizer.k):
                eigenface = recognizer.eigenfaces[:, i].reshape(height, width)
                # Normalize for better visualization
                eigenface_norm = (eigenface - eigenface.min()) / (eigenface.max() - eigenface.min())
                ax.imshow(eigenface_norm, cmap='gray')
                ax.set_title(f"Eigenface {i+1}", fontsize=10)
            ax.axis('off')
        
        plt.suptitle(f"Top {min(9, recognizer.k)} Eigenfaces", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
    except ImportError:
        print("Matplotlib not installed - skipping visualization")
        print("To install: pip install matplotlib")
    
    # Final summary
    print("\n" + "="*60)
    print("EXPERIMENT COMPLETE")
    print("="*60)
    print(f"\nTraining: First 9 images of each person ({num_persons * 9} total)")
    print(f"Testing: 10th image of {person_name}")
    print(f"Result: {'Recognized correctly' if y_pred[0] == y_test_true else 'Not recognized correctly'}")
    print(f"Distance to closest match: {distances[0]:.2f}")
    print(f"Eigenfaces used: {recognizer.k} (retained 95% variance)")

# --------------------------------------------------
# Run the experiment
# --------------------------------------------------
if __name__ == "__main__":
    main()
