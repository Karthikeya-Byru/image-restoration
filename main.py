from deoldify.visualize import *
import torch

# Needed for PyTorch >= 2.6 compatibility
import functools
import torch.serialization
torch.serialization.add_safe_globals([functools.partial, slice])

# Check GPU
torch.backends.cudnn.benchmark = True

# Initialize the colorizer
colorizer = get_image_colorizer(artistic=True)

# Path to your black & white image
image_path = r"C:\Users\Karthikeyan\Documents\Image-Restortation\DeOldify\test_images\pexels-jhawley-57905.jpg"

# Run colorization
colorizer.plot_transformed_image(
    path=image_path,
    render_factor=35,     # Try values from 20 to 40 for best quality
    compare=True          # Shows before/after comparison
)
