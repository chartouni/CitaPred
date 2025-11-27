"""
Neural network models for citation prediction using PyTorch.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


class CitationNeuralNet(nn.Module):
    """
    Feedforward neural network for citation prediction.
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: list = [128, 64, 32],
        dropout_rate: float = 0.3
    ):
        """
        Initialize the neural network.

        Args:
            input_size: Number of input features
            hidden_sizes: List of hidden layer sizes
            dropout_rate: Dropout probability for regularization
        """
        super(CitationNeuralNet, self).__init__()

        self.input_size = input_size
        self.hidden_sizes = hidden_sizes

        # Build layers
        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size

        # Output layer (single value for regression)
        layers.append(nn.Linear(prev_size, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x: Input tensor

        Returns:
            Output predictions
        """
        return self.network(x).squeeze()


class NeuralNetworkModel:
    """
    Wrapper class for PyTorch neural network with sklearn-like interface.
    """

    def __init__(
        self,
        hidden_sizes: list = [128, 64, 32],
        dropout_rate: float = 0.3,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        epochs: int = 100,
        early_stopping_patience: int = 10,
        device: Optional[str] = None
    ):
        """
        Initialize the neural network model.

        Args:
            hidden_sizes: List of hidden layer sizes
            dropout_rate: Dropout probability
            learning_rate: Learning rate for optimizer
            batch_size: Batch size for training
            epochs: Maximum number of training epochs
            early_stopping_patience: Epochs to wait before early stopping
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
        """
        self.hidden_sizes = hidden_sizes
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience

        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.model = None
        self.scaler = StandardScaler()
        self.input_size = None
        self.training_history = {'train_loss': [], 'val_loss': []}

        logger.info(f"Neural network initialized on device: {self.device}")

    def _prepare_data(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        fit_scaler: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Prepare data for training/prediction.

        Args:
            X: Feature matrix
            y: Target values (optional)
            fit_scaler: Whether to fit the scaler

        Returns:
            Tuple of (X_tensor, y_tensor)
        """
        # Scale features
        if fit_scaler:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)

        # Convert to tensors
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)

        if y is not None:
            y_tensor = torch.FloatTensor(y).to(self.device)
            return X_tensor, y_tensor
        else:
            return X_tensor, None

    def fit(self, X: np.ndarray, y: np.ndarray, validation_split: float = 0.2):
        """
        Train the neural network.

        Args:
            X: Training feature matrix
            y: Training target values
            validation_split: Fraction of data to use for validation
        """
        logger.info(f"Training neural network with {len(X)} samples")

        # Store input size and initialize model
        self.input_size = X.shape[1]
        self.model = CitationNeuralNet(
            input_size=self.input_size,
            hidden_sizes=self.hidden_sizes,
            dropout_rate=self.dropout_rate
        ).to(self.device)

        # Split data into train and validation
        val_size = int(len(X) * validation_split)
        train_size = len(X) - val_size

        indices = np.random.permutation(len(X))
        train_indices = indices[val_size:]
        val_indices = indices[:val_size]

        X_train, y_train = X[train_indices], y[train_indices]
        X_val, y_val = X[val_indices], y[val_indices]

        # Prepare data
        X_train_tensor, y_train_tensor = self._prepare_data(X_train, y_train, fit_scaler=True)
        X_val_tensor, y_val_tensor = self._prepare_data(X_val, y_val, fit_scaler=False)

        # Create data loaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Training loop with early stopping
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(self.epochs):
            # Training phase
            self.model.train()
            train_losses = []

            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                predictions = self.model(batch_X)
                loss = criterion(predictions, batch_y)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())

            avg_train_loss = np.mean(train_losses)

            # Validation phase
            self.model.eval()
            with torch.no_grad():
                val_predictions = self.model(X_val_tensor)
                val_loss = criterion(val_predictions, y_val_tensor).item()

            # Store history
            self.training_history['train_loss'].append(avg_train_loss)
            self.training_history['val_loss'].append(val_loss)

            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model state
                best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1

            # Log progress every 10 epochs
            if (epoch + 1) % 10 == 0:
                logger.info(
                    f"Epoch {epoch + 1}/{self.epochs} - "
                    f"Train Loss: {avg_train_loss:.4f}, "
                    f"Val Loss: {val_loss:.4f}"
                )

            # Early stopping
            if patience_counter >= self.early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                self.model.load_state_dict(best_model_state)
                break

        # Load best model
        if patience_counter < self.early_stopping_patience:
            self.model.load_state_dict(best_model_state)

        logger.info(f"Training completed. Best validation loss: {best_val_loss:.4f}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.

        Args:
            X: Feature matrix

        Returns:
            Predicted citation counts
        """
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")

        self.model.eval()
        with torch.no_grad():
            X_tensor, _ = self._prepare_data(X, fit_scaler=False)
            predictions = self.model(X_tensor)
            return predictions.cpu().numpy()

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate R² score.

        Args:
            X: Feature matrix
            y: True citation counts

        Returns:
            R² score
        """
        predictions = self.predict(X)
        ss_res = np.sum((y - predictions) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def get_training_history(self) -> dict:
        """
        Get training history.

        Returns:
            Dictionary with training and validation losses
        """
        return self.training_history
