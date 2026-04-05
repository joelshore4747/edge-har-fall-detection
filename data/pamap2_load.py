# data/pamap2_load.py
import pandas as pd
import glob

files = glob.glob("data/PAMAP2_Dataset/*.txt")
dfs = [pd.read_csv(f, sep=' ', header=None) for f in files]
df = pd.concat(dfs, ignore_index=True)

print("PAMAP2 total shape:", df.shape)
print(df.head())