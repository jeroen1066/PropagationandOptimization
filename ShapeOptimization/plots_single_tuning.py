import pickle
import matplotlib.pyplot as plt
import numpy as np
import math


fig, axs = plt.subplots(1,2)

labels = ['optimisation']

optimizer_name = 'sga'
data_file = 'results/' + optimizer_name + '_single_optimum.dat'
file = open(data_file,'rb')
data = pickle.load(file)
file.close()
for j in range(len(labels)):
    results = data[j]
    x = results[0]
    y = results[1]
    for i in range(len(x)):
        print(x[i])
    print('optimum')
    print(x[y.tolist().index(min(y))])
    axs[0].scatter(y,np.arange(len(y)), label=str(labels[j]))
axs[0].grid()
axs[0].set_xlabel('mass fitness')
axs[0].set_ylabel('population')
axs[0].set_ylim([0, 70])
axs[0].set_xlim([0, 8])
axs[0].legend()



for j in range(len(labels)):
    results = data[j]
    x = results[0]
    y = results[1]
    y_per_gen = results[2]
    y_arr = np.array(y_per_gen)
    avg_arr = np.zeros(y_arr.shape[0])
    for k in range(y_arr.shape[0]):
        gen_avg = 0
        num_valid = 0
        for l in range(y_arr.shape[1]):

            avg_one_pop = np.sum(y_arr[k,l])
            if avg_one_pop < 1000:
                gen_avg += avg_one_pop
                num_valid += 1
        avg_arr[k] = gen_avg/num_valid

    #average_fitness = np.sum(y_arr,axis=1)/3

    axs[1].plot(avg_arr, label=str(labels[j]))
axs[1].grid()
axs[1].set_xlabel('evolutions done')
axs[1].set_ylabel('Average fitness')
axs[1].legend()
plt.tight_layout()
plt.show()
