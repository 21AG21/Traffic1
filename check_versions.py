import platform
import numpy as np
import pandas as pd
import sklearn
import lightgbm as lgb
import torch

print("machine      ", platform.machine())
print("pandas       ", pd.__version__)
print("numpy        ", np.__version__)
print("scikit-learn ", sklearn.__version__)
print("lightgbm     ", lgb.__version__)
print("torch        ", torch.__version__)
print("GPU (mps)    ", torch.backends.mps.is_available())
