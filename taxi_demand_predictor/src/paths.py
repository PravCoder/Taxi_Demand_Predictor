from pathlib import Path
import os

# commonly used paths are stored here to import from
PARENT_DIR = Path(__file__).parent.resolve().parent
DATA_DIR = PARENT_DIR / 'data'
RAW_DATA_DIR = PARENT_DIR / 'data' / 'raw'   # path to the raw data directory
TRANSFORMED_DATA_DIR = PARENT_DIR / 'data' / 'transformed'
DATA_CACHE_DIR = PARENT_DIR / 'data' / 'cache'

MODELS_DIR = PARENT_DIR / 'models'

if not Path(DATA_DIR).exists():
    os.mkdir(DATA_DIR)

if not Path(RAW_DATA_DIR).exists():
    os.mkdir(RAW_DATA_DIR)

if not Path(TRANSFORMED_DATA_DIR).exists():
    os.mkdir(TRANSFORMED_DATA_DIR)

if not Path(MODELS_DIR).exists():
    os.mkdir(MODELS_DIR)

if not Path(DATA_CACHE_DIR).exists():
    os.mkdir(DATA_CACHE_DIR)


# in the data/raw folder all of the pulled-raw data is stored, where each file is data for a specific year and month
# in the data/transformed folder is the time-series and validated data