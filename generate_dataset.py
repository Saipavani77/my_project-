import pandas as pd
import numpy as np

# Generate a synthetic dataset
data = {
    'followers': np.random.randint(50, 1000, size=1000),
    'following': np.random.randint(50, 1000, size=1000),
    'engagement_rate': np.random.rand(1000),
    'is_promotional': np.random.choice([True, False], size=1000),
    'post_frequency': np.random.randint(1, 30, size=1000),
    'sentiment': np.random.uniform(-1, 1, size=1000),
    'category': np.random.choice(['influencer', 'brand', 'individual'], size=1000)
}

df = pd.DataFrame(data)
df.to_csv('user_data.csv', index=False)
