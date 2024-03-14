import numpy as np

#LDA is a dimensionality reduction technique and is often used for classification.
class LDA():
    def __init__(self, n_components):
        #The constructor initializes the number of components
        self.n_components = n_components
    
    def fit(self, X, y):
        
        n_samples, n_features = X.shape
        class_labels = np.unique(y)

        # Calculate overall mean
        mean_overall = np.mean(X, axis=0)
        self.mean_ = mean_overall  # Set the mean_ attribute

        # Between-class scatter matrix
        S_B = self._compute_between_class_scatter(X, y, mean_overall)

        # Within-class scatter matrix
        S_W = self._compute_within_class_scatter(X, y)

        # Solve the generalized eigenvalue problem SW^-1 SB
        A = np.linalg.inv(S_W) @ S_B
        eigenvalues, eigenvectors = np.linalg.eigh(A)

        # Sort eigenvectors by eigenvalues in descending order
        idxs = np.argsort(abs(eigenvalues))[::-1]
        eigenvectors = eigenvectors[:, idxs]

        # Select the first n_components eigenvectors
        self.L = eigenvectors[:, :self.n_components]

        # Compute variance explained ratio
        self.explained_variance_ratio = np.sum(eigenvalues[:self.n_components]) / np.sum(eigenvalues)
        
    def transform(self, X):
        X_centered = X - self.mean_
        print(X_centered.shape, self.L.shape)
        F = X_centered @ self.L
        return F

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)
    

    def _compute_between_class_scatter(self, X, y, mean_overall):
        n_components = X.shape[1]
        S_B = np.zeros((n_components, n_components))
        for c in np.unique(y):
            X_c = X[y == c]
            mean_c = np.mean(X_c, axis=0)
            n_c = X_c.shape[0]
            mean_diff = (mean_c - mean_overall).reshape(n_components, 1)
            S_B += n_c * (mean_diff @ mean_diff.T)
        return S_B

    def _compute_within_class_scatter(self, X, y):
        n_components = X.shape[1]
        S_W = np.zeros((n_components, n_components))
        for c in np.unique(y):
            X_c = X[y == c]
            mean_c = np.mean(X_c, axis=0)
            S_W += (X_c - mean_c).T @ (X_c - mean_c)
        return S_W