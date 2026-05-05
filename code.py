import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import time

# ==================================================
# GLOBAL SETTINGS
# ==================================================
PERSON_TO_TEST = "s41"
VARIANCE_RETAINED = 0.95

# ==================================================
# MANUAL LINEAR ALGEBRA FUNCTIONS
# ==================================================

def manual_mean(X):
    """Mean of columns"""
    d, N = X.shape
    mean = np.zeros((d, 1))
    for i in range(d):
        total = 0.0
        for j in range(N):
            total += X[i, j]
        mean[i, 0] = total / N
    return mean

def manual_center(X, mean):
    """Center data"""
    return X - mean

def manual_matmul(A, B):
    """Matrix multiplication"""
    m, n = A.shape
    n2, p = B.shape
    if n != n2:
        raise ValueError("Dimension mismatch")
    C = np.zeros((m, p))
    for i in range(m):
        for j in range(p):
            total = 0.0
            for k in range(n):
                total += A[i, k] * B[k, j]
            C[i, j] = total
    return C

def manual_transpose(A):
    """Matrix transpose"""
    m, n = A.shape
    At = np.zeros((n, m))
    for i in range(m):
        for j in range(n):
            At[j, i] = A[i, j]
    return At

def manual_norm(v):
    """Vector norm"""
    total = 0.0
    for val in v.flatten():
        total += val ** 2
    return np.sqrt(total)

def manual_normalize(v):
    """Normalize vector"""
    norm = manual_norm(v)
    return v / norm if norm > 1e-10 else v

def manual_power_iteration(A, max_iter=100):
    """Largest eigenvalue/eigenvector via power iteration"""
    n = A.shape[0]
    v = np.random.rand(n)
    v = v / manual_norm(v)

    for _ in range(max_iter):
        Av = manual_matmul(A, v.reshape(-1, 1)).flatten()
        v_new = Av / manual_norm(Av)
        if manual_norm(v_new - v) < 1e-8:
            return np.dot(Av, v) / np.dot(v, v), v_new
        v = v_new

    return np.dot(manual_matmul(A, v.reshape(-1, 1)).flatten(), v) / np.dot(v, v), v

def manual_svd_full(A):
    """Compute FULL SVD using Gram matrix trick"""
    N = A.shape[1]
    print(f"    Computing full SVD with {N} components...")
    
    # Gram matrix G = A^T * A (size N x N)
    At = manual_transpose(A)
    G = manual_matmul(At, A)
    
    # Get ALL eigenvalues and eigenvectors of G
    eigenvalues = []
    eigenvectors = []
    current_G = G.copy()
    
    for i in range(N):
        eigval, eigvec = manual_power_iteration(current_G)
        eigenvalues.append(eigval)
        eigenvectors.append(eigvec)
        
        # Deflation
        outer = np.outer(eigvec, eigvec)
        current_G = current_G - eigval * outer
        
        if (i + 1) % 50 == 0 or (i + 1) == N:
            print(f"        Computed {i+1}/{N} components...")
    
    eigenvalues = np.array(eigenvalues)
    V = np.array(eigenvectors).T
    
    # Singular values are sqrt of eigenvalues
    singular_values = np.sqrt(np.maximum(eigenvalues, 0))
    
    # U = A * V * S^{-1}
    S_inv = np.diag(1.0 / (singular_values + 1e-10))
    U = manual_matmul(A, V)
    U = manual_matmul(U, S_inv)
    
    # Normalize U columns
    for i in range(U.shape[1]):
        U[:, i:i+1] = manual_normalize(U[:, i:i+1])
    
    return U, singular_values, manual_transpose(V)

def manual_distance(a, b):
    """Euclidean distance"""
    diff = a - b
    total = 0.0
    for val in diff.flatten():
        total += val ** 2
    return np.sqrt(total)

# ==================================================
# DATA LOADING
# ==================================================

def load_dataset(dataset_path="dataset", images_per_person=10):
    """Load images from s1, s2, ... folders"""
    images = []
    labels = []
    person_names = []
    img_shape = None
    
    # Get all folders
    all_folders = sorted([f for f in os.listdir(dataset_path) 
                         if os.path.isdir(os.path.join(dataset_path, f))])
    
    # Filter to keep only s21 and above
    # folders = [f for f in all_folders if int(f[1:]) >= 21]

    # Train on the fuul set of data
    folders = all_folders
    
    print(f"Found {len(folders)} persons (from {folders[0]} to {folders[-1]})")
    
    for label, folder in enumerate(folders):
        folder_path = os.path.join(dataset_path, folder)
        person_names.append(folder)
        
        for i in range(1, images_per_person + 1):
            img_path = None
            for ext in ['.pgm', '.png', '.jpg']:
                test_path = os.path.join(folder_path, f"{i}{ext}")
                if os.path.exists(test_path):
                    img_path = test_path
                    break
            
            if img_path:
                img = Image.open(img_path).convert('L')
                if img_shape is None:
                    img_shape = img.size[::-1]
                images.append(np.array(img).flatten())
                labels.append(label)
    
    X = np.array(images).T
    y = np.array(labels)
    
    print(f"Loaded {X.shape[1]} images, each {X.shape[0]} pixels")
    print(f"Image shape: {img_shape[0]}x{img_shape[1]}")
    
    return X, y, person_names, img_shape

# ==================================================
# MAIN
# ==================================================

def main():
    start_time = time.time()
    
    print("="*60)
    print("EIGENFACE RECOGNITION")
    print(f"Testing: {PERSON_TO_TEST} (10th photo)")
    print("="*60)
    
    # Load data
    print("\n[1] Loading dataset...")
    X_all, y_all, person_names, img_shape = load_dataset()
    
    num_persons = len(person_names)
    imgs_per_person = 10
    
    # Training: first 9 images per person
    print("[2] Preparing training data (first 9 images)...")
    X_train = []
    y_train = []
    
    for pid in range(num_persons):
        start = pid * imgs_per_person
        for i in range(9):
            X_train.append(X_all[:, start + i])
            y_train.append(y_all[start + i])
    
    X_train = np.array(X_train).T
    y_train = np.array(y_train)
    
    print(f"    Training: {X_train.shape[1]} images")
    
    # Test: 10th image of specified person
    print(f"[3] Preparing test image (10th photo of {PERSON_TO_TEST})...")
    
    if PERSON_TO_TEST not in person_names:
        print(f"ERROR: {PERSON_TO_TEST} not found!")
        print(f"Available persons: {person_names}")
        return
    
    person_id = person_names.index(PERSON_TO_TEST)
    test_idx = person_id * imgs_per_person + 9
    X_test = X_all[:, test_idx:test_idx + 1]
    y_true = y_all[test_idx]
    
    # Compute mean face
    print("[4] Computing mean face...")
    mean_face = manual_mean(X_train)
    
    # Center data
    print("[5] Centering data...")
    A = manual_center(X_train, mean_face)
    
    # Compute FULL SVD (all components)
    print("[6] Computing FULL SVD...")
    U, S, Vt = manual_svd_full(A)
    
    # Calculate variance and select k based on 95%
    eigenvalues = S ** 2
    total_var = np.sum(eigenvalues)
    cum_var = np.cumsum(eigenvalues) / total_var
    
    # Find k such that cumulative variance >= VARIANCE_RETAINED
    k = 1
    for i, var in enumerate(cum_var):
        if var >= VARIANCE_RETAINED:
            k = i + 1
            break
    
    print(f"\n    Total components available: {len(S)}")
    print(f"    Selected k = {k} (retains {cum_var[k-1]*100:.2f}% variance)")
    
    # Keep only first k components
    U_k = U[:, :k]
    
    # Project training images
    print("[7] Projecting training images...")
    U_t = manual_transpose(U_k)
    train_proj = manual_matmul(U_t, A)
    
    # Recognize test image
    print("[8] Recognizing test image...")
    test_centered = manual_center(X_test, mean_face)
    test_proj = manual_matmul(U_t, test_centered)
    
    # Find nearest neighbor
    min_dist = float('inf')
    min_idx = -1
    
    for i in range(train_proj.shape[1]):
        dist = manual_distance(test_proj, train_proj[:, i:i+1])
        if dist < min_dist:
            min_dist = dist
            min_idx = i
    
    y_pred = y_train[min_idx]
    predicted_person = person_names[y_pred]
    
    # Results
    print("\n" + "="*60)
    print("RESULT")
    print("="*60)
    print(f"True: {PERSON_TO_TEST}")
    print(f"Predicted: {predicted_person}")
    
    if y_pred == y_true:
        print("\n✓ SUCCESS!")
    else:
        print("\n✗ FAILURE!")
    
    print(f"\nDistance to closest match: {min_dist:.4f}")
    
    # Reconstruct test image
    print("\n[9] Reconstructing test image...")
    reconstructed = manual_matmul(U_k, test_proj) + mean_face
    recon_error = manual_distance(X_test, reconstructed)
    print(f"Reconstruction error: {recon_error:.4f}")
    
    # Visualizations
    print("\n[10] Displaying results...")
    
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    
    # Mean face
    axes[0, 0].imshow(mean_face.reshape(img_shape), cmap='gray')
    axes[0, 0].set_title("Mean Face")
    axes[0, 0].axis('off')
    
    # Test image
    axes[0, 1].imshow(X_test.reshape(img_shape), cmap='gray')
    axes[0, 1].set_title(f"Test: {PERSON_TO_TEST}")
    axes[0, 1].axis('off')
    
    # Closest match
    closest = X_train[:, min_idx:min_idx+1]
    axes[0, 2].imshow(closest.reshape(img_shape), cmap='gray')
    axes[0, 2].set_title(f"Closest: {predicted_person}\nDist: {min_dist:.1f}")
    axes[0, 2].axis('off')
    
    # Reconstructed
    axes[1, 0].imshow(reconstructed.reshape(img_shape), cmap='gray')
    axes[1, 0].set_title(f"Reconstructed (k={k})")
    axes[1, 0].axis('off')
    
    # Error
    axes[1, 1].imshow(np.abs(X_test - reconstructed).reshape(img_shape), cmap='hot')
    axes[1, 1].set_title(f"Error: {recon_error:.2f}")
    axes[1, 1].axis('off')
    
    # Variance plot
    axes[1, 2].plot(range(1, len(cum_var)+1), cum_var*100, 'b-', linewidth=2)
    axes[1, 2].axhline(y=VARIANCE_RETAINED*100, color='r', linestyle='--', label='95% threshold')
    axes[1, 2].axvline(x=k, color='g', linestyle='--', label=f'k={k}')
    axes[1, 2].set_xlabel("Components (k)")
    axes[1, 2].set_ylabel("Cumulative Variance (%)")
    axes[1, 2].set_title(f"k={k}, {cum_var[k-1]*100:.1f}% variance")
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Show top 9 eigenfaces
    fig2, axes2 = plt.subplots(3, 3, figsize=(9, 9))
    for i in range(min(9, k)):
        row, col = i // 3, i % 3
        eigen = U_k[:, i].reshape(img_shape)
        eigen = (eigen - eigen.min()) / (eigen.max() - eigen.min() + 1e-10)
        axes2[row, col].imshow(eigen, cmap='gray')
        axes2[row, col].set_title(f"Eigenface {i+1}")
        axes2[row, col].axis('off')
    
    for i in range(min(9, k), 9):
        row, col = i // 3, i % 3
        axes2[row, col].axis('off')
    
    plt.suptitle(f"Top {min(9, k)} Eigenfaces")
    plt.tight_layout()
    plt.show()
    
    # Summary
    elapsed = time.time() - start_time
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Persons: {person_names[0]} to {person_names[-1]} ({num_persons} persons)")
    print(f"Training: {num_persons} × 9 = {X_train.shape[1]} images")
    print(f"Testing: 10th photo of {PERSON_TO_TEST}")
    print(f"Original dimension: {X_train.shape[0]} pixels")
    print(f"Reduced dimension (k): {k}")
    print(f"Variance retained: {cum_var[k-1]*100:.2f}%")
    print(f"Result: {'CORRECT' if y_pred == y_true else 'WRONG'}")
    print(f"Time: {elapsed:.2f} seconds")
    print("="*60)

if __name__ == "__main__":
    main()
