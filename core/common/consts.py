from pathlib import Path

# project structure
CONFIGS_DIR = Path('configs')
RUNS_DIR = Path('runs')
CORE_DIR = Path('core')

# file extensions
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')
NUMPY_EXTENSIONS = ('.npy',)
PICKLE_EXTENSIONS = ('.pickle',)
TORCH_EXTENSIONS = ('.pth', '.pt')
PYTHON_EXTENSIONS = ('.py',)

# misc
TIMESTAMP_FORMAT = "%d_%m_%y_at_%H_%M"
