# pytest runs this file first: it puts the project root on the import path so `import Src` works
# without installing anything.
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
