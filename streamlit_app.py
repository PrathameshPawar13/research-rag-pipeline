import os
import sys

os.environ["MPLBACKEND"] = "Agg"
import matplotlib
matplotlib.use("Agg")

sys.path.insert(0, os.path.dirname(__file__))

from ui.main import *
