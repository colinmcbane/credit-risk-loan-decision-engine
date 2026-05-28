# =============================================================================
# PHASE 3 — FEATURE ENGINEERING
# Credit Risk & Loan Decision Engine
# Author: Colin McBane
# Input: data/processed/sba_clean.db (built in Phase 1)
# Output: data/processed/feature_matrix.csv (input for Phase 4)
# =============================================================================

import pandas as pd
import numpy as np
import sqlite3
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA

# =============================================================================
# FILE PATHS
# =============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "data", "processed", "sba_clean.db")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "processed")
CHARTS_DIR = os.path.join(BASE_DIR, "outputs", "eda_charts")

os.makedirs(CHARTS_DIR, exist_ok=True)

print("=== Phase 3: Feature Engineering ===")
print(f"Reading from: {DB_PATH}")
print(f"Charts saved to: {CHARTS_DIR}")