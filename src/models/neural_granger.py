import torch
import torch.nn as nn
import numpy as np
from scipy import stats

class LSTMPredictor(nn.Module):
    """LSTM model for non-linear time-series prediction"""
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, 
                           dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        lstm_out, _ = self.lstm(x)
        # Use last hidden state for prediction
        predictions = self.fc(lstm_out[:, -1, :])
        return predictions

class NeuralGrangerTest:
    def __init__(self, max_lag=5):
        self.max_lag = max_lag

    def _create_sequences(self, target_var, data, source_var=None):
        """
        Create sequences for LSTM training.
        target_var: index of the target variable
        data: pandas DataFrame or numpy array
        source_var: index of the source variable (optional)
        """
        data_values = data.values if hasattr(data, 'values') else data
        n_samples, n_vars = data_values.shape
        
        X, y = [], []
        
        for i in range(self.max_lag, n_samples):
            # Target history
            target_hist = data_values[i-self.max_lag:i, target_var].reshape(-1, 1)
            
            if source_var is not None:
                # Source history
                source_hist = data_values[i-self.max_lag:i, source_var].reshape(-1, 1)
                # Combine
                seq = np.hstack([target_hist, source_hist])
            else:
                seq = target_hist
                
            X.append(seq)
            y.append(data_values[i, target_var])
            
        return np.array(X), np.array(y)
    
    def _train_model(self, X, y, epochs=50, device='cpu', hidden_dim=32):
        """Train LSTM and return final MSE loss"""
        input_dim = X.shape[2]
        model = LSTMPredictor(input_dim, hidden_dim=hidden_dim).to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
        X_tensor = torch.FloatTensor(X).to(device)
        y_tensor = torch.FloatTensor(y).view(-1, 1).to(device)
        
        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = model(X_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()
            
        model.eval()
        with torch.no_grad():
            final_preds = model(X_tensor)
            final_loss = criterion(final_preds, y_tensor).item()
        
        return final_loss 
    
    def granger_test(self, target_var, source_var, data, device='cpu'):
        """Test if source_var Granger-causes target_var"""
        
        # 1. Restricted Model (Target history only)
        X1, y = self._create_sequences(target_var, data) 
        loss1 = self._train_model(X1, y, device=device)
        
        # 2. Unrestricted Model (Target + Source history)
        X2, _ = self._create_sequences(target_var, data, source_var) 
        loss2 = self._train_model(X2, y, device=device)
        
        # 3. F-test on MSE improvement
        n = len(y)
        k_diff = X2.shape[2] - X1.shape[2] # Difference in complexity/parameters (input dim diff)
        
        # Note: This F-test is an approximation for neural networks. 
        # Strictly speaking, degrees of freedom in NNs are hard to define.
        # We use the input dimension difference as a proxy for added parameters relevant to the source.
        
        # Avoid division by zero
        if loss2 == 0:
            return 0.0, True

        # F-statistic formula: F = ((Loss_1 - Loss_2) / df_diff) / (Loss_2 / df_unrestricted)
        # df_unrestricted = n - p_unrestricted. We approximate p_unrestricted with input_dim * hidden etc?
        # For simplicity and robustness in this context, we often use a simpler ratio test or the standard regression F-test form
        # assuming the effective parameters added is proportional to k_diff.
        
        # Let's stick to the formula provided in the prompt but be careful with df.
        # The prompt uses: n - X2.shape[2] as denominator df. X2.shape[2] is input features per step.
        
        df1 = k_diff
        df2 = n - X2.shape[2]
        
        if df2 <= 0:
             return 1.0, False # Not enough data

        f_stat = ((loss1 - loss2) / df1) / (loss2 / df2)
        
        # If loss2 > loss1 (adding source made it worse), f_stat is negative -> p_value ~ 1
        if f_stat < 0:
            p_value = 1.0
        else:
            p_value = 1 - stats.f.cdf(f_stat, df1, df2)
        
        causes = p_value < 0.05
        return float(p_value), bool(causes)
