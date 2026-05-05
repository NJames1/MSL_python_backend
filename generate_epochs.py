import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, accuracy_score

# 1. LOAD YOUR REAL DATA
# Replace with your actual exported CSV filename
df = pd.read_csv('real_mamlaka_aw_data.csv') 

# Separate features (X) and target labels (y)
# Assuming 'location_id' is your target column
y = df['location_id']
X = df.drop(columns=['location_id'])

# 2. DIFFERENTIAL RSSI PREPROCESSING (Crucial for your methodology!)
# Replace missing values with -100 (standard for missing Wi-Fi)
X = X.fillna(-100)
# Apply your Differential RSSI equation: RSSI_diff = RSSI_i - max(RSSI_scan)
max_rssi_per_row = X.max(axis=1)
X_diff = X.apply(lambda row: row - max_rssi_per_row, axis=0)

# 3. TRAIN/TEST SPLIT & SCALING
X_train, X_test, y_train, y_test = train_test_split(X_diff, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. INITIALIZE THE SMALL NEURAL NETWORK
mlp = MLPClassifier(
    hidden_layer_sizes=(64, 32), 
    activation='relu', 
    solver='adam', 
    random_state=42,
    warm_start=True, # Allows us to train epoch by epoch
    max_iter=1       # Force 1 epoch per loop
)

# 5. EPOCH TRACKING LOOP
epochs = 150
train_loss, val_loss = [], []
train_acc, val_acc = [], []

classes = np.unique(y)

print("Training SNN on REAL data...")
for epoch in range(epochs):
    # Train for one epoch
    mlp.partial_fit(X_train_scaled, y_train, classes=classes)
    
    # Predict probabilities for loss calculation
    y_train_prob = mlp.predict_proba(X_train_scaled)
    y_test_prob = mlp.predict_proba(X_test_scaled)
    
    # Predict classes for accuracy calculation
    y_train_pred = mlp.predict(X_train_scaled)
    y_test_pred = mlp.predict(X_test_scaled)
    
    # Record metrics
    train_loss.append(log_loss(y_train, y_train_prob))
    val_loss.append(log_loss(y_test, y_test_prob))
    train_acc.append(accuracy_score(y_train, y_train_pred))
    val_acc.append(accuracy_score(y_test, y_test_pred))


# Removed dpi=300 from here so it displays normally on your screen
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot 1: Loss Curve
ax1.plot(train_loss, label='Training Loss', color='blue', linewidth=2)
ax1.plot(val_loss, label='Validation Loss', color='red', linestyle='--', linewidth=2)
ax1.set_title('SNN Convergence: Log Loss vs Epochs', fontsize=14)
ax1.set_xlabel('Epochs', fontsize=12)
ax1.set_ylabel('Log Loss', fontsize=12)
ax1.tick_params(axis='both', labelsize=10)
ax1.legend(fontsize=12)
ax1.grid(True, linestyle=':', alpha=0.7)

# Plot 2: Accuracy Curve
ax2.plot(train_acc, label='Training Accuracy', color='blue', linewidth=2)
ax2.plot(val_acc, label='Validation Accuracy', color='green', linestyle='--', linewidth=2)
ax2.set_title('SNN Convergence: Classification Accuracy vs Epochs', fontsize=14)
ax2.set_xlabel('Epochs', fontsize=12)
ax2.set_ylabel('Accuracy', fontsize=12)
ax2.tick_params(axis='both', labelsize=10)
ax2.legend(fontsize=12)
ax2.grid(True, linestyle=':', alpha=0.7)

# Add padding to tight_layout to prevent any title/axis overlap
plt.tight_layout(pad=3.0)

# Apply the high DPI ONLY when saving the image for your thesis
plt.savefig('SNN_Real_Data_Convergence.png', dpi=300, bbox_inches='tight')
print("✅ Plot saved successfully as 'SNN_Real_Data_Convergence.png'")

# Display the appropriately scaled version on your screen
plt.show()