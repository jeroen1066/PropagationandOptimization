import pickle
import matplotlib.pyplot as plt
import numpy as np
import math

# from optimization_tuning_v1_maco import seeds_to_test,generations_to_test,pops_to_test,kernel_to_test,convergence_speed_to_test,threshold_to_test,std_convergence_speed_to_test,eval_stop_to_test,focus_to_test

seeds_to_test = [42, 84, 144, 169, 74]
generations_to_test = [10, 20, 30, 40, 50]
pops_to_test = [30, 50, 75, 100, 125]

kernel_to_test = [10, 15, 20, 25, 30]
convergence_speed_to_test = [0.5, 1, 1.5, 2, 2.5]
threshold_to_test = [1, 2,  5, 10, 20]
std_convergence_speed_to_test = [3, 5, 7, 9, 11]
eval_stop_to_test = [10, 15, 20, 25, 30]
focus_to_test = [0.1,  0.3,  0.5, 0.7, 1.0]

optimizer_name = ['maco']

save_directory = 'ShapeOptimization/results/tuning_v2_maco/'

plot_seed = False
plot_generations = False
plot_pops = False

plot_kernel = False
plot_convergence_speed = False
plot_threshold = False
plot_std_convergence_speed = False
plot_eval_stop = False
plot_focus = False

if plot_seed:
    fig, axs = plt.subplots(2,2, figsize=(8,6))
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
    
    # plt.show()
    plt.tight_layout()
    plt.savefig('seed_tuning.png')

if plot_generations:
    fig, axs = plt.subplots(2,2, figsize=(8,6))
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
    # plt.show()
    plt.tight_layout()
    plt.savefig('generations_tuning.png')

if plot_pops:
    fig, axs = plt.subplots(2,2, figsize=(8,6))
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
    # plt.show()
    plt.tight_layout()
    plt.savefig('pops_tuning.png')

if plot_kernel:
    fig, axs = plt.subplots(2,2, figsize=(8,6))
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
    # plt.show()
    plt.tight_layout()
    plt.savefig('kernel_tuning.png')

if plot_convergence_speed:
    fig, axs = plt.subplots(2,2, figsize=(8,6))
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
    # plt.show()
    plt.tight_layout()
    plt.savefig('convergence_speed_tuning.png')

if plot_threshold:
    fig, axs = plt.subplots(2,2, figsize=(8,6))
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
    # plt.show()
    plt.tight_layout()
    plt.savefig('threshold_tuning.png')

if plot_std_convergence_speed:
    fig, axs = plt.subplots(2,2, figsize=(8,6))
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
    # plt.show()
    plt.tight_layout()
    plt.savefig('std_convergence_speed_tuning.png')

if plot_eval_stop:
    fig, axs = plt.subplots(2,2, figsize=(8,6))
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
    # plt.show()
    plt.tight_layout()
    plt.savefig('eval_stop_tuning.png')

if plot_focus:
    fig, axs = plt.subplots(2,2, figsize=(8,6))
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
    # plt.show()
    plt.tight_layout()
    plt.savefig('focus_tuning.png')

plot_simulation_speed = True

if plot_simulation_speed:
    # fig, axs = plt.subplots(2,2, figsize=(8,6))
    plt.figure(figsize=(8,5))
    x_values = np.arange(0, len(seeds_to_test))
    seeds_arr = np.zeros(len(seeds_to_test))
    generations_arr = np.zeros(len(generations_to_test))
    pops_arr = np.zeros(len(pops_to_test))
    kernel_arr = np.zeros(len(kernel_to_test))
    convergence_speed_arr = np.zeros(len(convergence_speed_to_test))
    threshold_arr = np.zeros(len(threshold_to_test))
    std_convergence_speed_arr = np.zeros(len(std_convergence_speed_to_test))
    eval_stop_arr = np.zeros(len(eval_stop_to_test))
    focus_arr = np.zeros(len(focus_to_test))

    include_seeds = True
    include_generations = True
    include_pops = True
    include_kernel = True
    include_convergence_speed = True
    include_threshold = False
    include_std_convergence_speed = True
    include_eval_stop = True
    include_focus = True

    if include_seeds:
        for i in range(len(seeds_to_test)):
            ancillary_file = save_directory + 'maco_seed_' + str(seeds_to_test[i]) + '_ancillary.txt'
            time_sim = np.genfromtxt(ancillary_file)
            seeds_arr[i] = time_sim
        plt.plot(x_values, seeds_arr, label='seeds', marker='o', linestyle='--')
    
    if include_generations:
        for i in range(len(generations_to_test)):
            ancillary_file = save_directory + 'maco_gens_' + str(generations_to_test[i]) + '_ancillary.txt'
            time_sim = np.genfromtxt(ancillary_file)
            generations_arr[i] = time_sim
        plt.plot(x_values, generations_arr, label='generations', marker='o', linestyle='--')
    
    if include_pops:
        for i in range(len(pops_to_test)):
            ancillary_file = save_directory + 'maco_pops_' + str(pops_to_test[i]) + '_ancillary.txt'
            time_sim = np.genfromtxt(ancillary_file)
            pops_arr[i] = time_sim
        plt.plot(x_values, pops_arr, label='pops', marker='o', linestyle='--')
    
    if include_kernel:
        for i in range(len(kernel_to_test)):
            ancillary_file = save_directory + 'maco_kernel_' + str(kernel_to_test[i]) + '_ancillary.txt'
            time_sim = np.genfromtxt(ancillary_file)
            kernel_arr[i] = time_sim
        plt.plot(x_values, kernel_arr, label='kernel', marker='o', linestyle='--')

    if include_convergence_speed:
        for i in range(len(convergence_speed_to_test)):
            ancillary_file = save_directory + 'maco_q_' + str(convergence_speed_to_test[i]) + '_ancillary.txt'
            time_sim = np.genfromtxt(ancillary_file)
            convergence_speed_arr[i] = time_sim
        plt.plot(x_values, convergence_speed_arr, label='conv. speed', marker='o', linestyle='--')
    
    if include_threshold:
        for i in range(len(threshold_to_test)):
            ancillary_file = save_directory + 'maco_threshold_' + str(threshold_to_test[i]) + '_ancillary.txt'
            time_sim = np.genfromtxt(ancillary_file)
            threshold_arr[i] = time_sim
        plt.plot(x_values, threshold_arr, label='threshold', marker='o', linestyle='--')
    
    if include_std_convergence_speed:
        for i in range(len(std_convergence_speed_to_test)):
            ancillary_file = save_directory + 'maco_n_gen_mark_' + str(std_convergence_speed_to_test[i]) + '_ancillary.txt'
            time_sim = np.genfromtxt(ancillary_file)
            std_convergence_speed_arr[i] = time_sim
        plt.plot(x_values, std_convergence_speed_arr, label='std conv. speed', marker='o', linestyle='--')
    
    if include_eval_stop:
        for i in range(len(eval_stop_to_test)):
            ancillary_file = save_directory + 'maco_evalstop_' + str(eval_stop_to_test[i]) + '_ancillary.txt'
            time_sim = np.genfromtxt(ancillary_file)
            eval_stop_arr[i] = time_sim
        plt.plot(x_values, eval_stop_arr, label='eval stop', marker='o', linestyle='--')

    if include_focus:
        for i in range(len(focus_to_test)):
            ancillary_file = save_directory + 'maco_focus_' + str(focus_to_test[i]) + '_ancillary.txt'
            time_sim = np.genfromtxt(ancillary_file)
            focus_arr[i] = time_sim
        plt.plot(x_values, focus_arr, label='focus', marker='o', linestyle='--')

    plt.grid()
    plt.xlabel('Parameter tuning index')
    plt.ylabel('Simulation time (s)')


    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.yscale('log')
    plt.xticks(np.arange(len(x_values)), x_values.astype(int))
    plt.show()
    # plt.savefig('simulation_speed_tuning.png')

        