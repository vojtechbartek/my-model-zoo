import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

DATA_FILE_PATH = os.getenv("DATA_FILE_PATH")
"""Path to data folder"""