import numpy as np 
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import pickle 
from matplotlib import style
import time


style.use('ggplot')

size = 10
hmEpisodes = 25000
movePenalty = 1
enemyPenalty = 300
foodReward = 25

epsilon = 0.9
EPS_DECAY = 0.9998

SHOW_EVERY = 3000

