import pandas as pd
import numpy as np

data = {
    'feature1': np.random.rand(100),
    'feature2': np.random.rand(100),
    'target': np.random.rand(100)
}

df = pd.DataFrame(data)

df.to_csv('data/synthetic_data.csv', index=False)
