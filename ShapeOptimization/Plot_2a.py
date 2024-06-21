import pickle
import matplotlib.pyplot as plt
import numpy as np
import math



optimizer_names = ['ihs', 'nsga2', 'moead', 'moead_gen', 'maco', 'nspso']

fig, axs = plt.subplots(2,3)

for i in range(6):
    optimizer_name = optimizer_names[i]
    data_file = 'results/' + optimizer_name + '.dat'
    file = open(data_file,'rb')
    data = pickle.load(file)
    file.close()
    for j in range(4):
        results = data[j]
        x = results[0]
        y = results[1]
        axs[math.floor(i/3),i%3].scatter(y[0],y[1], label='generation '+str(j+1))
    axs[math.floor(i/3),i%3].grid()
    axs[math.floor(i/3),i%3].set_xlabel('mass fitness')
    axs[math.floor(i/3),i%3].set_ylabel('l/d fitness')
    axs[math.floor(i/3),i%3].set_ylim([0,8])
    axs[math.floor(i/3),i%3].set_xlim([0,8])
    axs[math.floor(i/3),i%3].legend()
plt.show()
    
fig, axs = plt.subplots(2,3)
for i in range(6):
    optimizer_name = optimizer_names[i]
    data_file = 'results/' + optimizer_name + '.dat'
    file = open(data_file,'rb')
    data = pickle.load(file)
    file.close()
    for j in range(4):
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

                avg_one_pop = np.sum(y_arr[k,l])/3
                if avg_one_pop < 1000:
                    gen_avg += avg_one_pop
                    num_valid += 1
            avg_arr[k] = gen_avg/num_valid

        #average_fitness = np.sum(y_arr,axis=1)/3

        axs[math.floor(i/3),i%3].plot(avg_arr, label='generation '+str(j+1))
    axs[math.floor(i/3),i%3].grid()
    axs[math.floor(i/3),i%3].set_xlabel('evolutions done')
    axs[math.floor(i/3),i%3].set_ylabel('Average fitness')
    axs[math.floor(i/3),i%3].legend()
plt.show()