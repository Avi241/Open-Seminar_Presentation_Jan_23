from ctypes import sizeof
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import math

df = pd.read_csv('xandy.csv')
#print(df)
Array2d_result = df.to_numpy()
print(Array2d_result.shape)

x1, y1 = Array2d_result.T
# plt.subplot(2, 1, 1)
# plt.plot(x1, y1)
# plt.show()
angle = 220
theta = (angle/180.) * np.pi

rotMatrix = np.array([[np.cos(theta), -np.sin(theta)], 
                         [np.sin(theta),  np.cos(theta)]])

m = np.dot(rotMatrix,[x1, y1])
print(m.shape)
x1, y1 = m
# plt.subplot(2, 1, 2)
plt.figure(figsize = (18, 5.5), dpi=80)
plt.plot(x1, y1-1, label = 'Test Trajectory')
plt.xticks(np.arange(0, max(x1),1))
plt.yticks(np.arange(0, max(y1), 1))
# axis_scale = 2
# x1min, x1max = plt.xlim()
# y1min, y1max = plt.ylim()
plt.tick_params(axis='y', which = 'major', labelsize = 7)
# plt.xlim(x1min*axis_scale, x1max*axis_scale)
# plt.ylim(y1min*axis_scale, y1max*axis_scale)
dataGT = np.array([[0,0],
[9,0], [16.8,0],
[16.8,4.8],[19.2,4.8],[19.2, 6], [17.4,6], [17.4, 4.8], [16.8, 4.8], [16.8, 4.2], [8.7, 4.2], [8.7, 0], [0,0]])

x2, y2 = dataGT.T
plt.plot(x2*1.15, y2, label = "Ground Truth")
plt.xticks(np.arange(0, max(x1),1))
plt.yticks(np.arange(0, max(y1),1))
plt.legend()
plt.xlabel("x axis (in metres)")
plt.ylabel("y axis (in metres)")
plt.title("vins_mono_SP14_test")
plt.show()
