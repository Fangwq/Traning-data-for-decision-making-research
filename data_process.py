import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json


file = "best.json"
with open(file, 'r') as load_file:
    load_dict = json.load(load_file)
    N = len(load_dict)
    data = load_dict[0]
    print(data)                # each element: state(length: 29), action(length: 3), reward(length: 1)
    lstate = len(data[0])
    laction = len(data[1])
    lreward = len(np.array([data[2]]))
    state = np.zeros([N, lstate])
    action = np.zeros([N, laction])
    reward = np.zeros([N, lreward])
    for i in range(N):
        state[i] = load_dict[i][0]
        action[i] = load_dict[i][1]
        reward[i] = load_dict[i][2]

print(np.shape(state))
print(np.shape(action))

plt.figure()
plt.plot(xrange(N), action[:,0], "r.-", label = "steer")
plt.figure()
plt.plot(xrange(N), action[:,1], "r.-", label = "accelerate")
plt.figure()
plt.plot(xrange(N), action[:,2], "r.-", label = "brake")
plt.show()