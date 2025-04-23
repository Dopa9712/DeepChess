import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import and run main with command line arguments
from src.main import main
import sys

if __name__ == "__main__":
    main()