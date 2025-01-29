import torch
import torch.nn as nn
from tqdm import tqdm

class RidgeClassificationHead(nn.Module):
    def __init__(self, input_dim, num_classes, alpha=1.0, bias=True):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.alpha = alpha
        self.linear = nn.Linear(input_dim, num_classes, bias=bias)
        self.linear.requires_grad_(False)
        
    def compute_class_weights(self, y):
        """
        Compute balanced class weights similar to sklearn's implementation:
        w_j = n_samples / (n_classes * n_samples_j)
        """
        n_samples = len(y)
        n_classes = self.num_classes
        
        # Count samples per class
        class_counts = torch.bincount(y, minlength=n_classes)
        
        # Compute weights (n_samples / (n_classes * n_samples_j))
        class_weights = n_samples / (n_classes * class_counts.float())
        
        # Handle division by zero (if any class has zero samples)
        class_weights = torch.nan_to_num(class_weights, nan=1.0, posinf=1.0, neginf=1.0)
        
        return class_weights
    
    def fit(self, X, y):
        """
        Fit Ridge classifier with balanced class weights.
        Args:
            X: input features [n_samples, n_features]
            y: class labels [n_samples]
        """
        # Compute balanced class weights
        class_weights = self.compute_class_weights(y)
        
        # Create sample weights matrix based on class weights
        sample_weights = class_weights[y]  # [n_samples]
        
        # Reshape sample weights for broadcasting
        sample_weights = sample_weights.view(-1, 1)  # [n_samples, 1]

        # Augment X with a column of ones to include the bias term
        X_augmented = torch.cat([X, torch.ones((X.shape[0], 1), device=X.device)], dim=1)  # [n_samples, n_features + 1]
        
        # Apply weights to augmented X and y_one_hot
        y_one_hot = torch.nn.functional.one_hot(y, self.num_classes) # [n_samples, n_classes]
        weighted_X_augmented = X_augmented * torch.sqrt(sample_weights)  # [n_samples, n_features + 1]
        weighted_y = y_one_hot * torch.sqrt(sample_weights)  # [n_samples, n_classes]
        
        # Compute the closed-form solution for weighted Ridge Regression
        # W = (X^T W X + αI)^(-1) X^T W Y
        eps = 1e-6
        identity = torch.eye(X_augmented.shape[1], device=X.device)
        
        XTX = torch.mm(weighted_X_augmented.t(), weighted_X_augmented)
        reg_term = self.alpha * identity
        inverse_term = torch.linalg.inv(XTX + reg_term + eps * identity)
        XTY = torch.mm(weighted_X_augmented.t(), weighted_y)
        W_augmented = torch.mm(inverse_term, XTY)  # [n_features + 1, n_classes]
        
        # Separate weights and bias
        W = W_augmented[:-1, :]  # [n_features, n_classes]
        bias = W_augmented[-1, :]  # [n_classes]
        
        # Set the weights and bias
        self.linear.weight.copy_(W.t())
        if self.linear.bias is not None:
            self.linear.bias.copy_(bias)
    
    def forward(self, x):
        return self.linear(x)
    
def get_ridge_classification_head(image_encoder, val_dataloader, num_classes, device, alpha=1.0, bias=True):

    image_encoder.to(device)
    image_encoder.eval()
    
    # Collect features and labels
    X, y = [], []
    print("\nCollecting features for classification head...")
    with torch.no_grad():
        for images, labels in tqdm(val_dataloader):
            images = images.to(device)
            features = image_encoder(images).cpu()
            X.append(features)
            y.append(labels)
    
    X = torch.cat(X, dim=0).to(device)
    y = torch.cat(y, dim=0).to(device)
    
    # Create and fit the Ridge Classification Head
    input_dim = X.shape[1]
    ridge_head = RidgeClassificationHead(input_dim, num_classes, alpha=alpha, bias=bias).to(device)
    ridge_head.fit(X, y)
    
    return ridge_head