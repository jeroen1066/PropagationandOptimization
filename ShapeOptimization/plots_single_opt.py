import pickle
import matplotlib.pyplot as plt
import numpy as np
import math



optimizer_names = ['de', 'sade', 'de1220', 'gwo', 'ihs', 'pso'] #, 'pso_gen', 'sea', 'sga', 'bee_colony']

fig, axs = plt.subplots(2,3)

for i in range(len(optimizer_names)):
    optimizer_name = optimizer_names[i]
    data_file = 'results/' + optimizer_name + '_single.dat'
    file = open(data_file,'rb')
    data = pickle.load(file)
    file.close()
    for j in range(2):
        results = data[j]
        x = results[0]
        y = results[1]
        print(len(y))
        axs[math.floor(i/3),i%3].scatter(y[0],len(y[0]), label='generation '+str(j+1))
    axs[math.floor(i / 3), i % 3].grid()
    axs[math.floor(i / 3), i % 3].set_xlabel('mass fitness')
    axs[math.floor(i / 3), i % 3].set_ylabel('l/d fitness')
    axs[math.floor(i / 3), i % 3].set_ylim([0, 8])
    axs[math.floor(i / 3), i % 3].set_xlim([0, 8])
    axs[math.floor(i / 3), i % 3].legend()
plt.show()