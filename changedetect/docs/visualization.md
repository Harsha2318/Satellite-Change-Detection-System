# Visualization Module Documentation

## Overview

The visualization module provides functions for creating various visualizations related to satellite image change detection. These visualizations can be used for data exploration, model evaluation, and result presentation.

## Functions

### Display Functions

#### `display_image_pair`

Displays a pair of satellite images side by side for visual comparison.

```python
from changedetect.src.utils.visualization import display_image_pair

display_image_pair(
    image1, 
    image2, 
    titles=["Before", "After"],
    figsize=(12, 6),
    cmap="viridis",
    alpha=1.0
)
```

**Parameters:**
- `image1`: First image (numpy array)
- `image2`: Second image (numpy array)
- `titles`: List of titles for the images
- `figsize`: Figure size (width, height) in inches
- `cmap`: Colormap to use for grayscale images
- `alpha`: Transparency level (0.0 to 1.0)

#### `display_change_detection_results`

Displays the results of change detection alongside the original images.

```python
from changedetect.src.utils.visualization import display_change_detection_results

display_change_detection_results(
    image1,
    image2,
    prediction,
    ground_truth=None,
    titles=["Before", "After", "Prediction", "Ground Truth"],
    figsize=(16, 4),
    cmap="viridis",
    alpha=0.7
)
```

**Parameters:**
- `image1`: First image (numpy array)
- `image2`: Second image (numpy array)
- `prediction`: Predicted change mask (binary numpy array)
- `ground_truth`: Ground truth change mask (binary numpy array)
- `titles`: List of titles for the images
- `figsize`: Figure size (width, height) in inches
- `cmap`: Colormap to use for grayscale images
- `alpha`: Transparency level for overlays

### Overlay Functions

#### `create_change_overlay`

Creates an overlay of change mask on top of an image.

```python
from changedetect.src.utils.visualization import create_change_overlay

overlay = create_change_overlay(
    image,
    mask,
    color=[1, 0, 0],  # Red
    alpha=0.5
)
```

**Parameters:**
- `image`: Base image (numpy array)
- `mask`: Binary mask to overlay (numpy array)
- `color`: RGB color for the overlay
- `alpha`: Transparency level (0.0 to 1.0)

**Returns:**
- `overlay`: Combined image with mask overlay

#### `create_multi_class_overlay`

Creates an overlay of multiple class masks on an image, each with a different color.

```python
from changedetect.src.utils.visualization import create_multi_class_overlay

overlay = create_multi_class_overlay(
    image,
    masks,
    colors=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],  # Red, Green, Blue
    alpha=0.5
)
```

**Parameters:**
- `image`: Base image (numpy array)
- `masks`: List of binary masks to overlay
- `colors`: List of RGB colors for each mask
- `alpha`: Transparency level (0.0 to 1.0)

**Returns:**
- `overlay`: Combined image with multiple mask overlays

### Evaluation Visualizations

#### `plot_confusion_matrix`

Creates a visual representation of a confusion matrix for change detection.

```python
from changedetect.src.utils.visualization import plot_confusion_matrix

plot_confusion_matrix(
    prediction,
    ground_truth,
    labels=["No Change", "Change"],
    figsize=(8, 8),
    cmap="Blues",
    normalize=True
)
```

**Parameters:**
- `prediction`: Predicted binary mask (numpy array)
- `ground_truth`: Ground truth binary mask (numpy array)
- `labels`: List of class names
- `figsize`: Figure size (width, height) in inches
- `cmap`: Colormap for the confusion matrix
- `normalize`: Whether to normalize the confusion matrix

#### `plot_precision_recall_curve`

Plots the precision-recall curve for change detection results.

```python
from changedetect.src.utils.visualization import plot_precision_recall_curve

plot_precision_recall_curve(
    prediction_prob,
    ground_truth,
    figsize=(8, 6)
)
```

**Parameters:**
- `prediction_prob`: Prediction probabilities (numpy array)
- `ground_truth`: Ground truth binary mask (numpy array)
- `figsize`: Figure size (width, height) in inches

#### `plot_roc_curve`

Plots the ROC curve for change detection results.

```python
from changedetect.src.utils.visualization import plot_roc_curve

plot_roc_curve(
    prediction_prob,
    ground_truth,
    figsize=(8, 6)
)
```

**Parameters:**
- `prediction_prob`: Prediction probabilities (numpy array)
- `ground_truth`: Ground truth binary mask (numpy array)
- `figsize`: Figure size (width, height) in inches

### Geospatial Visualizations

#### `plot_geospatial_changes`

Visualizes change detection results on a map with geospatial coordinates.

```python
from changedetect.src.utils.visualization import plot_geospatial_changes

plot_geospatial_changes(
    image,
    change_mask,
    transform,
    figsize=(10, 10),
    cmap="viridis",
    alpha=0.7,
    basemap=True
)
```

**Parameters:**
- `image`: Base satellite image (numpy array)
- `change_mask`: Change detection mask (binary numpy array)
- `transform`: Affine transform for geospatial coordinates
- `figsize`: Figure size (width, height) in inches
- `cmap`: Colormap for the image
- `alpha`: Transparency level for the change overlay
- `basemap`: Whether to include a basemap

### Animation Functions

#### `create_change_animation`

Creates an animation showing the transition between before and after images.

```python
from changedetect.src.utils.visualization import create_change_animation

create_change_animation(
    image1,
    image2,
    output_path="change_animation.gif",
    duration=2.0,
    fps=30
)
```

**Parameters:**
- `image1`: First image (numpy array)
- `image2`: Second image (numpy array)
- `output_path`: Path to save the animation
- `duration`: Duration of the animation in seconds
- `fps`: Frames per second

### Batch Visualization

#### `create_batch_visualization`

Creates a grid of visualizations for a batch of image pairs and their change detection results.

```python
from changedetect.src.utils.visualization import create_batch_visualization

create_batch_visualization(
    batch_images1,
    batch_images2,
    batch_predictions,
    batch_ground_truths=None,
    max_samples=16,
    figsize=(20, 20)
)
```

**Parameters:**
- `batch_images1`: Batch of first images (numpy array)
- `batch_images2`: Batch of second images (numpy array)
- `batch_predictions`: Batch of predictions (numpy array)
- `batch_ground_truths`: Batch of ground truth masks (numpy array)
- `max_samples`: Maximum number of samples to display
- `figsize`: Figure size (width, height) in inches

### Saving Functions

#### `save_visualization`

Saves the current visualization to a file.

```python
from changedetect.src.utils.visualization import save_visualization

save_visualization(
    output_path,
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.1
)
```

**Parameters:**
- `output_path`: Path to save the visualization
- `dpi`: Resolution in dots per inch
- `bbox_inches`: Bounding box in inches
- `pad_inches`: Padding in inches

## Usage Examples

### Basic Image Pair Visualization

```python
import numpy as np
from changedetect.src.utils.visualization import display_image_pair

# Load satellite images
image1 = np.random.rand(256, 256, 3)  # Before image
image2 = np.random.rand(256, 256, 3)  # After image

# Display the image pair
display_image_pair(image1, image2, titles=["2022-01-01", "2023-01-01"])
```

### Change Detection Results Visualization

```python
import numpy as np
from changedetect.src.utils.visualization import display_change_detection_results

# Load satellite images and masks
image1 = np.random.rand(256, 256, 3)  # Before image
image2 = np.random.rand(256, 256, 3)  # After image
prediction = (np.random.rand(256, 256) > 0.8).astype(np.uint8)  # Predicted changes
ground_truth = (np.random.rand(256, 256) > 0.9).astype(np.uint8)  # Ground truth

# Display the change detection results
display_change_detection_results(image1, image2, prediction, ground_truth)
```

### Creating and Saving a Complex Visualization

```python
import numpy as np
import matplotlib.pyplot as plt
from changedetect.src.utils.visualization import create_change_overlay, save_visualization

# Load image and change mask
image = np.random.rand(256, 256, 3)
mask = (np.random.rand(256, 256) > 0.9).astype(np.uint8)

# Create overlay
overlay = create_change_overlay(image, mask, color=[1, 0, 0], alpha=0.7)

# Create custom plot
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(overlay)
plt.title("Change Detection")
plt.axis("off")

plt.suptitle("Urban Development Analysis", fontsize=16)

# Save the visualization
save_visualization("urban_changes.png", dpi=300)
```

## Integration with Other Modules

The visualization module is designed to work seamlessly with other components of the change detection system:

- **Data Module**: Visualize raw satellite images and preprocessed data
- **Model Module**: Display model predictions and evaluation metrics
- **Postprocessing Module**: Compare raw and refined change detection results

## Customization

The visualization functions are built on top of matplotlib and can be customized by accessing the matplotlib figure and axes objects:

```python
import matplotlib.pyplot as plt
from changedetect.src.utils.visualization import display_image_pair

# Display image pair
fig, axes = display_image_pair(image1, image2, return_fig_axes=True)

# Customize the figure
fig.suptitle("Urbanization Analysis 2022-2023", fontsize=18)
for ax in axes:
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```