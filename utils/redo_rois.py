import os
import pandas as pd

files = os.listdir("rois")

df = pd.DataFrame()

for i, f in enumerate(files):
    d = pd.read_csv("rois/" + f)
    d["id"] = i
    df = df.append(d, ignore_index=True)

df.to_csv("rois.csv", index=False)