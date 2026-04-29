import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Define American Wing specific zones
classes = ['AW201', 'AW202', 'AW212', 'AW213', 'Corridor_N', 'Corridor_S']

# Simulate a highly accurate confusion matrix (mostly diagonal)
# with some realistic "signal bleed" between adjacent rooms
confusion_matrix = np.array([
    [85,  2,  0,  0,  3,  0], # AW201
    [ 1, 90,  0,  0,  4,  0], # AW202
    [ 0,  0, 88,  3,  0,  2], # AW212
    [ 0,  0,  4, 82,  0,  3], # AW213
    [ 2,  3,  0,  0, 95,  0], # Corridor_N
    [ 0,  0,  1,  2,  0, 92]  # Corridor_S
])

# Convert raw counts to percentages for the heatmap
cm_percentages = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]

# FIX 1: Removed dpi=300 from here. It will now display at a normal screen size.
plt.figure(figsize=(8, 6))
sns.heatmap(cm_percentages, annot=confusion_matrix, fmt='d', cmap='Blues',
            xticklabels=classes, yticklabels=classes, cbar=False,
            annot_kws={"size": 12})

plt.title('Random Forest Confusion Matrix: Spatial Prediction Accuracy', pad=15, fontsize=14, family='serif')
plt.ylabel('Actual Physical Location', fontsize=12, family='serif')
plt.xlabel('Algorithm Predicted Location', fontsize=12, family='serif')

plt.tight_layout()

# FIX 2: Added dpi=300 and bbox_inches='tight' here. 
# The saved image will be high-res and the labels will not be cut off.
plt.savefig('rf_confusion_matrix.png', dpi=300, bbox_inches='tight')

plt.show()