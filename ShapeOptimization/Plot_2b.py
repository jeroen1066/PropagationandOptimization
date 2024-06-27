import pickle
import matplotlib.pyplot as plt
import numpy as np
import math

# from optimization_tuning_v1_maco import seeds_to_test,generations_to_test,pops_to_test,kernel_to_test,convergence_speed_to_test,threshold_to_test,std_convergence_speed_to_test,eval_stop_to_test,focus_to_test

seeds_to_test = [42, 84, 144, 169, 74]
generations_to_test = [10, 15, 25, 30, 40, 50]
pops_to_test = [30, 50, 75, 100, 125]

kernel_to_test = [10, 15, 20, 25, 30]
convergence_speed_to_test = [0.5, 1, 1.5, 2, 2.5]
threshold_to_test = [1, 2,  5, 10, 20]
std_convergence_speed_to_test = [3, 5, 7, 9, 11]
eval_stop_to_test = [50, 75, 100, 125, 150]
focus_to_test = [0.1,  0.3,  0.5, 0.7, 1.0]

optimizer_name = ['maco']

save_directory = 'ShapeOptimization/results/tuning_v2_maco/'

plot_seed = True
plot_generations = True
plot_pops = True

plot_kernel = True
plot_convergence_speed = True
plot_threshold = False
plot_std_convergence_speed = True
plot_eval_stop = True
plot_focus = True



if plot_seed:
    fig, axs = plt.subplots(2,2)
    for i in range(len(seeds_to_test)):
        data_file = save_directory + 'maco_seed_' + str(seeds_to_test[i]) + '.dat'
        ancillary_file = save_directory + 'maco_seed_' + str(seeds_to_test[i]) + '_ancillary.txt'
        file = open(data_file,'rb')
        data = pickle.load(file)[0]
        file.close()
        x = data[0]
        y = data[1]
        y_per_gen = np.array(data[2])
        # plot 1
        axs[(0, 0)].scatter(np.asarray(y)[:,0], np.asarray(y)[:,1], label='seed ' + str(seeds_to_test[i]))
        axs[(0, 0)].set_ylim([0, 8])
        axs[(0, 0)].set_xlim([0, 8])
        axs[(0, 0)].grid()
        axs[(0, 0)].set_xlabel('mass fitness')
        axs[(0, 0)].set_ylabel('l/d fitness')
        axs[(0, 0)].legend()
        # plot 2
        axs[(0, 1)].scatter(np.asarray(y)[:, 0], np.asarray(y)[:, 2], label='seed ' + str(seeds_to_test[i]))
        axs[(0, 1)].set_ylim([0, 8])
        axs[(0, 1)].set_xlim([0, 8])
        axs[(0, 1)].grid()
        axs[(0, 1)].set_xlabel('mass fitness')
        axs[(0, 1)].set_ylabel('g-load fitness')
        axs[(0, 1)].legend()
        # plot 3
        axs[(1, 0)].scatter(np.asarray(y)[:, 1], np.asarray(y)[:, 2], label='seed ' + str(seeds_to_test[i]))
        axs[(1, 0)].set_ylim([0, 8])
        axs[(1, 0)].set_xlim([0, 8])
        axs[(1, 0)].grid()
        axs[(1, 0)].set_xlabel('l/d fitness')
        axs[(1, 0)].set_ylabel('g-load fitness')
        axs[(1, 0)].legend()
        # plot 3
        avg_arr = np.zeros(y_per_gen.shape[0])
        for k in range(y_per_gen.shape[0]):
            gen_avg = 0
            num_valid = 0
            for l in range(y_per_gen.shape[1]):
                avg_one_pop = np.sum(y_per_gen[k,l])/3
                if avg_one_pop < 1000:
                    gen_avg += avg_one_pop
                    num_valid += 1
            avg_arr[k] = gen_avg / num_valid

        time_sim = np.genfromtxt(ancillary_file)
        # plt.scatter(y[0],y[1],label='Seed '+str(seeds_to_test[i]))
        #average_fitness = np.sum(y_arr,axis=1)/3

        axs[(1, 1)].plot(avg_arr, label='seed ' + str(seeds_to_test[i]))
        axs[(1, 1)].grid()
        axs[(1, 1)].set_xlabel('evolutions done')
        axs[(1, 1)].set_ylabel('average fitness')
        axs[(1, 1)].legend()
    plt.show()

if plot_generations:
    fig, axs = plt.subplots(2,2)
    for i in range(len(generations_to_test)):
        data_file = save_directory + 'maco_gens_' + str(generations_to_test[i]) + '.dat'
        ancillary_file = save_directory + 'maco_gens_' + str(generations_to_test[i]) + '_ancillary.txt'
        file = open(data_file,'rb')
        data = pickle.load(file)[0]
        file.close()
        x = data[0]
        y = data[1]
        y_per_gen = np.array(data[2])
        # plot 1
        axs[(0, 0)].scatter(np.asarray(y)[:,0], np.asarray(y)[:,1], label=str(generations_to_test[i]) + ' generations')
        axs[(0, 0)].set_ylim([0, 8])
        axs[(0, 0)].set_xlim([0, 8])
        axs[(0, 0)].grid()
        axs[(0, 0)].set_xlabel('mass fitness')
        axs[(0, 0)].set_ylabel('l/d fitness')
        axs[(0, 0)].legend()
        # plot 2
        axs[(0, 1)].scatter(np.asarray(y)[:, 0], np.asarray(y)[:, 2], label=str(generations_to_test[i]) + ' generations')
        axs[(0, 1)].set_ylim([0, 8])
        axs[(0, 1)].set_xlim([0, 8])
        axs[(0, 1)].grid()
        axs[(0, 1)].set_xlabel('mass fitness')
        axs[(0, 1)].set_ylabel('g-load fitness')
        axs[(0, 1)].legend()
        # plot 3
        axs[(1, 0)].scatter(np.asarray(y)[:, 1], np.asarray(y)[:, 2], label=str(generations_to_test[i]) + ' generations')
        axs[(1, 0)].set_ylim([0, 8])
        axs[(1, 0)].set_xlim([0, 8])
        axs[(1, 0)].grid()
        axs[(1, 0)].set_xlabel('l/d fitness')
        axs[(1, 0)].set_ylabel('g-load fitness')
        axs[(1, 0)].legend()
        # plot 3
        avg_arr = np.zeros(y_per_gen.shape[0])
        for k in range(y_per_gen.shape[0]):
            gen_avg = 0
            num_valid = 0
            for l in range(y_per_gen.shape[1]):
                avg_one_pop = np.sum(y_per_gen[k,l])/3
                if avg_one_pop < 1000:
                    gen_avg += avg_one_pop
                    num_valid += 1
            avg_arr[k] = gen_avg / num_valid

        time_sim = np.genfromtxt(ancillary_file)
        # plt.scatter(y[0],y[1],label='Seed '+str(seeds_to_test[i]))
        #average_fitness = np.sum(y_arr,axis=1)/3

        axs[(1, 1)].plot(avg_arr, label= str(generations_to_test[i]) + ' generations')
        axs[(1, 1)].grid()
        axs[(1, 1)].set_xlabel('evolutions done')
        axs[(1, 1)].set_ylabel('average fitness')
        axs[(1, 1)].legend()
    plt.show()

if plot_pops:
    fig, axs = plt.subplots(2,2)
    for i in range(len(pops_to_test)):
        data_file = save_directory + 'maco_pops_' + str(pops_to_test[i]) + '.dat'
        ancillary_file = save_directory + 'maco_pops_' + str(pops_to_test[i]) + '_ancillary.txt'
        file = open(data_file,'rb')
        data = pickle.load(file)[0]
        file.close()
        x = data[0]
        y = data[1]
        y_per_gen = np.array(data[2])
        # plot 1
        axs[(0, 0)].scatter(np.asarray(y)[:,0], np.asarray(y)[:,1], label=str(pops_to_test[i]) + ' pops')
        axs[(0, 0)].set_ylim([0, 8])
        axs[(0, 0)].set_xlim([0, 8])
        axs[(0, 0)].grid()
        axs[(0, 0)].set_xlabel('mass fitness')
        axs[(0, 0)].set_ylabel('l/d fitness')
        axs[(0, 0)].legend()
        # plot 2
        axs[(0, 1)].scatter(np.asarray(y)[:, 0], np.asarray(y)[:, 2], label=str(pops_to_test[i]) + ' pops')
        axs[(0, 1)].set_ylim([0, 8])
        axs[(0, 1)].set_xlim([0, 8])
        axs[(0, 1)].grid()
        axs[(0, 1)].set_xlabel('mass fitness')
        axs[(0, 1)].set_ylabel('g-load fitness')
        axs[(0, 1)].legend()
        # plot 3
        axs[(1, 0)].scatter(np.asarray(y)[:, 1], np.asarray(y)[:, 2], label=str(pops_to_test[i]) + ' pops')
        axs[(1, 0)].set_ylim([0, 8])
        axs[(1, 0)].set_xlim([0, 8])
        axs[(1, 0)].grid()
        axs[(1, 0)].set_xlabel('l/d fitness')
        axs[(1, 0)].set_ylabel('g-load fitness')
        axs[(1, 0)].legend()
        # plot 3
        avg_arr = np.zeros(y_per_gen.shape[0])
        for k in range(y_per_gen.shape[0]):
            gen_avg = 0
            num_valid = 0
            for l in range(y_per_gen.shape[1]):
                avg_one_pop = np.sum(y_per_gen[k,l])/3
                if avg_one_pop < 1000:
                    gen_avg += avg_one_pop
                    num_valid += 1
            avg_arr[k] = gen_avg / num_valid
        
        time_sim = np.genfromtxt(ancillary_file)
        # plt.scatter(y[0],y[1],label='Seed '+str(seeds_to_test[i]))
        #average_fitness = np.sum(y_arr,axis=1)/3

        axs[(1, 1)].plot(avg_arr, label= str(pops_to_test[i]) + ' pop')
        axs[(1, 1)].grid()
        axs[(1, 1)].set_xlabel('evolutions done')
        axs[(1, 1)].set_ylabel('average fitness')
        axs[(1, 1)].legend()
    plt.show()

if plot_kernel:
    fig, axs = plt.subplots(2,2)
    for i in range(len(kernel_to_test)):
        data_file = save_directory + 'maco_kernel_' + str(kernel_to_test[i]) + '.dat'
        ancillary_file = save_directory + 'maco_kernel_' + str(kernel_to_test[i]) + '_ancillary.txt'
        file = open(data_file,'rb')
        data = pickle.load(file)[0]
        file.close()
        x = data[0]
        y = data[1]
        y_per_gen = np.array(data[2])
        # plot 1
        axs[(0, 0)].scatter(np.asarray(y)[:,0], np.asarray(y)[:,1], label=str(kernel_to_test[i]) + ' kernel')
        axs[(0, 0)].set_ylim([0, 8])
        axs[(0, 0)].set_xlim([0, 8])
        axs[(0, 0)].grid()
        axs[(0, 0)].set_xlabel('mass fitness')
        axs[(0, 0)].set_ylabel('l/d fitness')
        axs[(0, 0)].legend()
        # plot 2
        axs[(0, 1)].scatter(np.asarray(y)[:, 0], np.asarray(y)[:, 2], label=str(kernel_to_test[i]) + ' kernel')
        axs[(0, 1)].set_ylim([0, 8])
        axs[(0, 1)].set_xlim([0, 8])
        axs[(0, 1)].grid()
        axs[(0, 1)].set_xlabel('mass fitness')
        axs[(0, 1)].set_ylabel('g-load fitness')
        axs[(0, 1)].legend()
        # plot 3
        axs[(1, 0)].scatter(np.asarray(y)[:, 1], np.asarray(y)[:, 2], label=str(kernel_to_test[i]) + ' kernel')
        axs[(1, 0)].set_ylim([0, 8])
        axs[(1, 0)].set_xlim([0, 8])
        axs[(1, 0)].grid()
        axs[(1, 0)].set_xlabel('l/d fitness')
        axs[(1, 0)].set_ylabel('g-load fitness')
        axs[(1, 0)].legend()
        # plot 3
        avg_arr = np.zeros(y_per_gen.shape[0])
        for k in range(y_per_gen.shape[0]):
            gen_avg = 0
            num_valid = 0
            for l in range(y_per_gen.shape[1]):
                avg_one_pop = np.sum(y_per_gen[k,l])/3
                if avg_one_pop < 1000:
                    gen_avg += avg_one_pop
                    num_valid += 1
            avg_arr[k] = gen_avg / num_valid
        
        time_sim = np.genfromtxt(ancillary_file)
        # plt.scatter(y[0],y[1],label='Seed '+str(seeds_to_test[i]))
        #average_fitness = np.sum(y_arr,axis=1)/3

        axs[(1, 1)].plot(avg_arr, label= str(kernel_to_test[i]) + ' kernel')
        axs[(1, 1)].grid()
        axs[(1, 1)].set_xlabel('evolutions done')
        axs[(1, 1)].set_ylabel('average fitness')
        axs[(1, 1)].legend()
    plt.show()

if plot_convergence_speed:
    fig, axs = plt.subplots(2,2)
    for i in range(len(convergence_speed_to_test)):
        data_file = save_directory + 'maco_q_' + str(convergence_speed_to_test[i]) + '.dat'
        ancillary_file = save_directory + 'maco_q_' + str(convergence_speed_to_test[i]) + '_ancillary.txt'
        file = open(data_file,'rb')
        data = pickle.load(file)[0]
        file.close()
        x = data[0]
        y = data[1]
        y_per_gen = np.array(data[2])
        # plot 1
        axs[(0, 0)].scatter(np.asarray(y)[:,0], np.asarray(y)[:,1], label=str(convergence_speed_to_test[i]) + ' conv speed')
        axs[(0, 0)].set_ylim([0, 8])
        axs[(0, 0)].set_xlim([0, 8])
        axs[(0, 0)].grid()
        axs[(0, 0)].set_xlabel('mass fitness')
        axs[(0, 0)].set_ylabel('l/d fitness')
        axs[(0, 0)].legend()
        # plot 2
        axs[(0, 1)].scatter(np.asarray(y)[:, 0], np.asarray(y)[:, 2], label=str(convergence_speed_to_test[i]) + ' conv speed')
        axs[(0, 1)].set_ylim([0, 8])
        axs[(0, 1)].set_xlim([0, 8])
        axs[(0, 1)].grid()
        axs[(0, 1)].set_xlabel('mass fitness')
        axs[(0, 1)].set_ylabel('g-load fitness')
        axs[(0, 1)].legend()
        # plot 3
        axs[(1, 0)].scatter(np.asarray(y)[:, 1], np.asarray(y)[:, 2], label=str(convergence_speed_to_test[i]) + ' conv speed')
        axs[(1, 0)].set_ylim([0, 8])
        axs[(1, 0)].set_xlim([0, 8])
        axs[(1, 0)].grid()
        axs[(1, 0)].set_xlabel('l/d fitness')
        axs[(1, 0)].set_ylabel('g-load fitness')
        axs[(1, 0)].legend()
        # plot 3
        avg_arr = np.zeros(y_per_gen.shape[0])
        for k in range(y_per_gen.shape[0]):
            gen_avg = 0
            num_valid = 0
            for l in range(y_per_gen.shape[1]):
                avg_one_pop = np.sum(y_per_gen[k,l])/3
                if avg_one_pop < 1000:
                    gen_avg += avg_one_pop
                    num_valid += 1
            avg_arr[k] = gen_avg / num_valid
        
        time_sim = np.genfromtxt(ancillary_file)
        # plt.scatter(y[0],y[1],label='Seed '+str(seeds_to_test[i]))
        #average_fitness = np.sum(y_arr,axis=1)/3

        axs[(1, 1)].plot(avg_arr, label= str(convergence_speed_to_test[i]) + ' conv speed')
        axs[(1, 1)].grid()
        axs[(1, 1)].set_xlabel('evolutions done')
        axs[(1, 1)].set_ylabel('average fitness')
        axs[(1, 1)].legend()
    plt.show()

if plot_threshold:
    fig, axs = plt.subplots(2,2)
    for i in range(len(threshold_to_test)):
        data_file = save_directory + 'maco_threshold_' + str(threshold_to_test[i]) + '.dat'
        ancillary_file = save_directory + 'maco_threshold_' + str(threshold_to_test[i]) + '_ancillary.txt'
        file = open(data_file,'rb')
        data = pickle.load(file)[0]
        file.close()
        x = data[0]
        y = data[1]
        y_per_gen = np.array(data[2])
        # plot 1
        axs[(0, 0)].scatter(np.asarray(y)[:,0], np.asarray(y)[:,1], label=str(threshold_to_test[i]) + ' threshold')
        axs[(0, 0)].set_ylim([0, 8])
        axs[(0, 0)].set_xlim([0, 8])
        axs[(0, 0)].grid()
        axs[(0, 0)].set_xlabel('mass fitness')
        axs[(0, 0)].set_ylabel('l/d fitness')
        axs[(0, 0)].legend()
        # plot 2
        axs[(0, 1)].scatter(np.asarray(y)[:, 0], np.asarray(y)[:, 2], label=str(threshold_to_test[i]) + ' threshold')
        axs[(0, 1)].set_ylim([0, 8])
        axs[(0, 1)].set_xlim([0, 8])
        axs[(0, 1)].grid()
        axs[(0, 1)].set_xlabel('mass fitness')
        axs[(0, 1)].set_ylabel('g-load fitness')
        axs[(0, 1)].legend()
        # plot 3
        axs[(1, 0)].scatter(np.asarray(y)[:, 1], np.asarray(y)[:, 2], label=str(threshold_to_test[i]) + ' threshold')
        axs[(1, 0)].set_ylim([0, 8])
        axs[(1, 0)].set_xlim([0, 8])
        axs[(1, 0)].grid()
        axs[(1, 0)].set_xlabel('l/d fitness')
        axs[(1, 0)].set_ylabel('g-load fitness')
        axs[(1, 0)].legend()
        # plot 3
        avg_arr = np.zeros(y_per_gen.shape[0])
        for k in range(y_per_gen.shape[0]):
            gen_avg = 0
            num_valid = 0
            for l in range(y_per_gen.shape[1]):
                avg_one_pop = np.sum(y_per_gen[k,l])/3
                if avg_one_pop < 1000:
                    gen_avg += avg_one_pop
                    num_valid += 1
            avg_arr[k] = gen_avg / num_valid
        
        time_sim = np.genfromtxt(ancillary_file)
        # plt.scatter(y[0],y[1],label='Seed '+str(seeds_to_test[i]))
        #average_fitness = np.sum(y_arr,axis=1)/3

        axs[(1, 1)].plot(avg_arr, label= str(threshold_to_test[i]) + ' threshold')
        axs[(1, 1)].grid()
        axs[(1, 1)].set_xlabel('evolutions done')
        axs[(1, 1)].set_ylabel('average fitness')
        axs[(1, 1)].legend()
    plt.show()

if plot_std_convergence_speed:
    fig, axs = plt.subplots(2,2)
    for i in range(len(std_convergence_speed_to_test)):
        data_file = save_directory + 'maco_n_gen_mark_' + str(std_convergence_speed_to_test[i]) + '.dat'
        ancillary_file = save_directory + 'maco_n_gen_mark_' + str(std_convergence_speed_to_test[i]) + '_ancillary.txt'
        file = open(data_file,'rb')
        data = pickle.load(file)[0]
        file.close()
        x = data[0]
        y = data[1]
        y_per_gen = np.array(data[2])
        # plot 1
        axs[(0, 0)].scatter(np.asarray(y)[:,0], np.asarray(y)[:,1], label=str(std_convergence_speed_to_test[i]) + ' std conv speed')
        axs[(0, 0)].set_ylim([0, 8])
        axs[(0, 0)].set_xlim([0, 8])
        axs[(0, 0)].grid()
        axs[(0, 0)].set_xlabel('mass fitness')
        axs[(0, 0)].set_ylabel('l/d fitness')
        axs[(0, 0)].legend()
        # plot 2
        axs[(0, 1)].scatter(np.asarray(y)[:, 0], np.asarray(y)[:, 2], label=str(std_convergence_speed_to_test[i]) + ' std conv speed')
        axs[(0, 1)].set_ylim([0, 8])
        axs[(0, 1)].set_xlim([0, 8])
        axs[(0, 1)].grid()
        axs[(0, 1)].set_xlabel('mass fitness')
        axs[(0, 1)].set_ylabel('g-load fitness')
        axs[(0, 1)].legend()
        # plot 3
        axs[(1, 0)].scatter(np.asarray(y)[:, 1], np.asarray(y)[:, 2], label=str(std_convergence_speed_to_test[i]) + ' std conv speed')
        axs[(1, 0)].set_ylim([0, 8])
        axs[(1, 0)].set_xlim([0, 8])
        axs[(1, 0)].grid()
        axs[(1, 0)].set_xlabel('l/d fitness')
        axs[(1, 0)].set_ylabel('g-load fitness')
        axs[(1, 0)].legend()
        # plot 3
        avg_arr = np.zeros(y_per_gen.shape[0])
        for k in range(y_per_gen.shape[0]):
            gen_avg = 0
            num_valid = 0
            for l in range(y_per_gen.shape[1]):
                avg_one_pop = np.sum(y_per_gen[k,l])/3
                if avg_one_pop < 1000:
                    gen_avg += avg_one_pop
                    num_valid += 1
            avg_arr[k] = gen_avg / num_valid
        
        time_sim = np.genfromtxt(ancillary_file)
        # plt.scatter(y[0],y[1],label='Seed '+str(seeds_to_test[i]))
        #average_fitness = np.sum(y_arr,axis=1)/3

        axs[(1, 1)].plot(avg_arr, label= str(std_convergence_speed_to_test[i]) + ' std conv speed')
        axs[(1, 1)].grid()
        axs[(1, 1)].set_xlabel('evolutions done')
        axs[(1, 1)].set_ylabel('average fitness')
        axs[(1, 1)].legend()
    plt.show()

if plot_eval_stop:
    fig, axs = plt.subplots(2,2)
    for i in range(len(eval_stop_to_test)):
        data_file = save_directory + 'maco_evalstop_' + str(eval_stop_to_test[i]) + '.dat'
        ancillary_file = save_directory + 'maco_evalstop_' + str(eval_stop_to_test[i]) + '_ancillary.txt'
        file = open(data_file,'rb')
        data = pickle.load(file)[0]
        file.close()
        x = data[0]
        y = data[1]
        y_per_gen = np.array(data[2])
        # plot 1
        axs[(0, 0)].scatter(np.asarray(y)[:,0], np.asarray(y)[:,1], label=str(eval_stop_to_test[i]) + ' eval stop')
        axs[(0, 0)].set_ylim([0, 8])
        axs[(0, 0)].set_xlim([0, 8])
        axs[(0, 0)].grid()
        axs[(0, 0)].set_xlabel('mass fitness')
        axs[(0, 0)].set_ylabel('l/d fitness')
        axs[(0, 0)].legend()
        # plot 2
        axs[(0, 1)].scatter(np.asarray(y)[:, 0], np.asarray(y)[:, 2], label=str(eval_stop_to_test[i]) + ' eval stop')
        axs[(0, 1)].set_ylim([0, 8])
        axs[(0, 1)].set_xlim([0, 8])
        axs[(0, 1)].grid()
        axs[(0, 1)].set_xlabel('mass fitness')
        axs[(0, 1)].set_ylabel('g-load fitness')
        axs[(0, 1)].legend()
        # plot 3
        axs[(1, 0)].scatter(np.asarray(y)[:, 1], np.asarray(y)[:, 2], label=str(eval_stop_to_test[i]) + ' eval stop')
        axs[(1, 0)].set_ylim([0, 8])
        axs[(1, 0)].set_xlim([0, 8])
        axs[(1, 0)].grid()
        axs[(1, 0)].set_xlabel('l/d fitness')
        axs[(1, 0)].set_ylabel('g-load fitness')
        axs[(1, 0)].legend()
        # plot 3
        avg_arr = np.zeros(y_per_gen.shape[0])
        for k in range(y_per_gen.shape[0]):
            gen_avg = 0
            num_valid = 0
            for l in range(y_per_gen.shape[1]):
                avg_one_pop = np.sum(y_per_gen[k,l])/3
                if avg_one_pop < 1000:
                    gen_avg += avg_one_pop
                    num_valid += 1
            avg_arr[k] = gen_avg / num_valid
        
        time_sim = np.genfromtxt(ancillary_file)
        # plt.scatter(y[0],y[1],label='Seed '+str(seeds_to_test[i]))
        #average_fitness = np.sum(y_arr,axis=1)/3

        axs[(1, 1)].plot(avg_arr, label= str(eval_stop_to_test[i]) + ' eval stop')
        axs[(1, 1)].grid()
        axs[(1, 1)].set_xlabel('evolutions done')
        axs[(1, 1)].set_ylabel('average fitness')
        axs[(1, 1)].legend()
    plt.show()

if plot_focus:
    fig, axs = plt.subplots(2,2)
    for i in range(len(focus_to_test)):
        data_file = save_directory + 'maco_focus_' + str(focus_to_test[i]) + '.dat'
        ancillary_file = save_directory + 'maco_focus_' + str(focus_to_test[i]) + '_ancillary.txt'
        file = open(data_file,'rb')
        data = pickle.load(file)[0]
        file.close()
        x = data[0]
        y = data[1]
        y_per_gen = np.array(data[2])
        # plot 1
        axs[(0, 0)].scatter(np.asarray(y)[:,0], np.asarray(y)[:,1], label=str(focus_to_test[i]) + ' focus')
        axs[(0, 0)].set_ylim([0, 8])
        axs[(0, 0)].set_xlim([0, 8])
        axs[(0, 0)].grid()
        axs[(0, 0)].set_xlabel('mass fitness')
        axs[(0, 0)].set_ylabel('l/d fitness')
        axs[(0, 0)].legend()
        # plot 2
        axs[(0, 1)].scatter(np.asarray(y)[:, 0], np.asarray(y)[:, 2], label=str(focus_to_test[i]) + ' focus')
        axs[(0, 1)].set_ylim([0, 8])
        axs[(0, 1)].set_xlim([0, 8])
        axs[(0, 1)].grid()
        axs[(0, 1)].set_xlabel('mass fitness')
        axs[(0, 1)].set_ylabel('g-load fitness')
        axs[(0, 1)].legend()
        # plot 3
        axs[(1, 0)].scatter(np.asarray(y)[:, 1], np.asarray(y)[:, 2], label=str(focus_to_test[i]) + ' focus')
        axs[(1, 0)].set_ylim([0, 8])
        axs[(1, 0)].set_xlim([0, 8])
        axs[(1, 0)].grid()
        axs[(1, 0)].set_xlabel('l/d fitness')
        axs[(1, 0)].set_ylabel('g-load fitness')
        axs[(1, 0)].legend()
        # plot 3
        avg_arr = np.zeros(y_per_gen.shape[0])
        for k in range(y_per_gen.shape[0]):
            gen_avg = 0
            num_valid = 0
            for l in range(y_per_gen.shape[1]):
                avg_one_pop = np.sum(y_per_gen[k,l])/3
                if avg_one_pop < 1000:
                    gen_avg += avg_one_pop
                    num_valid += 1
            avg_arr[k] = gen_avg / num_valid
        
        time_sim = np.genfromtxt(ancillary_file)
        # plt.scatter(y[0],y[1],label='Seed '+str(seeds_to_test[i]))
        #average_fitness = np.sum(y_arr,axis=1)/3

        axs[(1, 1)].plot(avg_arr, label= str(focus_to_test[i]) + ' focus')
        axs[(1, 1)].grid()
        axs[(1, 1)].set_xlabel('evolutions done')
        axs[(1, 1)].set_ylabel('average fitness')
        axs[(1, 1)].legend()
    plt.show()
                           