import pickle
import matplotlib.pyplot as plt
import numpy as np
import math

# from optimization_tuning_v1_maco import seeds_to_test,generations_to_test,pops_to_test,kernel_to_test,convergence_speed_to_test,threshold_to_test,std_convergence_speed_to_test,eval_stop_to_test,focus_to_test

seeds_to_test = [42, 84, 144, 169, 74, 29, 60, 1745, 1480025]
generations_to_test = [25, 50, 75, 100, 150]
pops_to_test = [30, 50, 75, 100, 150]

kernel_to_test = [10, 30, 50, 70, 90]
convergence_speed_to_test = [0.5, 1, 1.5, 2, 2.5]
threshold_to_test = [1, 2,  5, 10, 20]
std_convergence_speed_to_test = [3, 5, 7, 9, 11, 15, 21]
eval_stop_to_test = [50000, 75000, 100000, 125000, 150000, 200000, 300000]
focus_to_test = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0]

optimizer_name = ['maco']

save_directory = 'ShapeOptimization/results/tuning_v1_maco/'

plot_seed = True
plot_generations = True
plot_pops = True

plot_kernel = True
plot_convergence_speed = True
plot_threshold = True
plot_std_convergence_speed = True
plot_eval_stop = True
plot_focus = True


if plot_seed:
    for i in range(len(seeds_to_test)):
        data_file = save_directory + 'maco_seed_' + str(seeds_to_test[i]) + '.dat'
        ancillary_file = save_directory + 'maco_seed_' + str(seeds_to_test[i]) + '_ancillary.txt'
        file = open(data_file,'rb')
        data = pickle.load(file)[0]
        file.close()
        x = data[0]
        y = data[1]
        time_sim = np.genfromtxt(ancillary_file)
        plt.scatter(y[0],y[1],label='Seed '+str(seeds_to_test[i]))

    plt.xlabel('Objective 1')
    plt.ylabel('Objective 2')
    plt.title('Seed')
    plt.legend()
    plt.show()

