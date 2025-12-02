import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import NearestNeighbors
import warnings
warnings.filterwarnings('ignore')

# Auto-reload modules
%load_ext autoreload
%autoreload 2

# Import project modules
from clustering_models import BrandClusterer
from dimensionality_reduction import DimensionalityReducer
from visualization_utils import ClusterVisualizer

# Import hyperparameter tuner
from clustering_models import HyperparameterTuner

# Set display options
pd.set_option('display.max_columns', None)
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)