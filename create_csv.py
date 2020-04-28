import os
import pandas as pd
from tqdm import tqdm
from imageio import imread

root = './gestures'

# the gesture id is directly extracted from the
# directory path and saved as the first column.
# example of directory path: "./gestures/0"
# id is available from the 11th index.

for directory, _, files in os.walk(root):
    data = []
    for file in tqdm(files):
        im = imread(os.path.join(directory, file))
        value = im.flatten()
        data.append([directory[11:]] + list(value))
    with open('train.csv', 'a') as f:
        pd.DataFrame(data).to_csv(f, header=False, index=False)
