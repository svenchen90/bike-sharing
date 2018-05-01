from data_normalization import rescaleNormalization
from data_load_helper import loadData
from GD import gradientDescent
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd

train = pd.read_csv('hour.csv')
print(train)