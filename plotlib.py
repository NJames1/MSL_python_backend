import matplotlib.pyplot as plt
import numpy as np

# Set plot style for academic paper
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({'font.size': 12, 'font.family': 'serif'})

# Simulate realistic SNN loss data over 50 epochs
epochs = np.arange(1, 51)
# Training loss drops sharply, then asymptotes
train_loss = 2.5 * np.exp(-0.15 * epochs) + 0.15 + np.random.normal(0, 0.02, 50)
# Validation loss follows closely but stays slightly higher
val_loss = 2.4 * np.exp(-0.13 * epochs) + 0.22 + np.random.normal(0, 0.03, 50)

# Smooth the curves slightly for a cleaner look
train_loss_smooth = np.convolve(train_loss, np.ones(3)/3, mode='same')
val_loss_smooth = np.convolve(val_loss, np.ones(3)/3, mode='same')

# FIX 1: Removed dpi=300 from here. It will now default to standard screen resolution.
fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(epochs[1:-1], train_loss_smooth[1:-1], label='Training Loss', color='#1f77b4', linewidth=2)
ax.plot(epochs[1:-1], val_loss_smooth[1:-1], label='Validation Loss', color='#ff7f0e', linewidth=2, linestyle='--')

ax.set_title('SNN Optimization: Training vs. Validation Loss (American Wing)', pad=15)
ax.set_xlabel('Training Epochs')
ax.set_ylabel('Categorical Cross-Entropy Loss')
ax.legend(loc='upper right', frameon=True, shadow=True)

plt.tight_layout()

# FIX 2: Added dpi=300 and bbox_inches='tight' here. 
# This ensures the saved image is high-res and nothing gets cut off in the file.
plt.savefig('snn_loss_curve.png', dpi=300, bbox_inches='tight')

# The display window will now be a normal, manageable size.
plt.show()