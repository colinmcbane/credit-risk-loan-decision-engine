# =============================================================================
# PHASE 1 — SQL DATA CLEANING
# Credit Risk & Loan Decision Engine
# Author: Colin McBane
# Data: SBA 7(a) FOIA Loan Data (FY2010-2019, FY2020-Present)
# =============================================================================

import pandas as pd
import sqlite3
import os
from tqdm import tqdm