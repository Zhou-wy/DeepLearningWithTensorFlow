# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import tensorflow as tf
import torch
import cv2
import numpy as np
import matplotlib as mlt
import pandas as pd

for moudle in tf, torch, np, mlt, pd, cv2:
    print(moudle.__name__, moudle.__version__)
