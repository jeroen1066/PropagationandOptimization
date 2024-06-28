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
plot_threshold = True
plot_std_convergence_speed = True
plot_eval_stop = True
plot_focus = True


if plot_seed:
    plt.figure()
    for i in range(len(seeds_to_test)):
        data_file = save_directory + 'maco_seed_' + str(seeds_to_test[i]) + '.dat'
        ancillary_file = save_directory + 'maco_seed_' + str(seeds_to_test[i]) + '_ancillary.txt'
        file = open(data_file,'rb')
        data = pickle.load(file)[0]
        file.close()
        x = data[0]
        y = data[1]
        y_per_gen = np.array(data[2])
        avg_arr = np.zeros(y_per_gen.shape[0])
        for k in range(y_per_gen.shape[0]):


            gen_avg = 0
            num_valid = 0
            for l in range(y_per_gen.shape[1]):
                avg_one_pop = np.sum(y_per_gen[k,l])/3
                if avg_one_pop < 1000:
                    gen_avg += avg_one_pop
                    num_valid += 1
            avg_arr[k] = gen_avg/num_valid


        time_sim = np.genfromtxt(ancillary_file)
        # plt.scatter(y[0],y[1],label='Seed '+str(seeds_to_test[i]))
        #average_fitness = np.sum(y_arr,axis=1)/3

        plt.plot(avg_arr, label='Seed: '+str(seeds_to_test[i]))

    plt.grid()
    plt.ylabel("Average Fitness")
    plt.xlabel("Generations")
    plt.legend()
    plt.show()

if plot_generations:
    plt.figure()
    for i in range(len(generations_to_test)):
        data_file = save_directory + 'maco_generations_' + str(generations_to_test[i]) + '.dat'
        ancillary_file = save_directory + 'maco_generations_' + str(generations_to_test[i]) + '_ancillary.txt'
        file = open(data_file,'rb')
        data = pickle.load(file)[0]
        file.close()
        x = data[0]
        y = data[1]
        y_per_gen = np.array(data[2])
        avg_arr = np.zeros(y_per_gen.shape[0])
        for k in range(y_per_gen.shape[0]):


            gen_avg = 0
            num_valid = 0
            for l in range(y_per_gen.shape[1]):
                avg_one_pop = np.sum(y_per_gen[k,l])/3
                if avg_one_pop < 1000:
                    gen_avg += avg_one_pop
                    num_valid += 1
            avg_arr[k] = gen_avg/num_valid


        time_sim = np.genfromtxt(ancillary_file)
        # plt.scatter(y[0],y[1],label='Seed '+str(seeds_to_test[i]))
        #average_fitness = np.sum(y_arr,axis=1)/3

        plt.plot(avg_arr, label='Generations: '+str(generations_to_test[i]))

    plt.grid()
    plt.ylabel("Average Fitness")
    plt.xlabel("Generations")
    plt.legend()
    plt.show()

if plot_pops:
    plt.figure()
    for i in range(len(pops_to_test)):
        data_file = save_directory + 'maco_pops_' + str(pops_to_test[i]) + '.dat'
        ancillary_file = save_directory + 'maco_pops_' + str(pops_to_test[i]) + '_ancillary.txt'
        file = open(data_file,'rb')
        data = pickle.load(file)[0]
        file.close()
        x = data[0]
        y = data[1]
        y_per_gen = np.array(data[2])
        avg_arr = np.zeros(y_per_gen.shape[0])
        for k in range(y_per_gen.shape[0]):


            gen_avg = 0
            num_valid = 0
            for l in range(y_per_gen.shape[1]):
                avg_one_pop = np.sum(y_per_gen[k,l])/3
                if avg_one_pop < 1000:
                    gen_avg += avg_one_pop
                    num_valid += 1
            avg_arr[k] = gen_avg/num_valid


        time_sim = np.genfromtxt(ancillary_file)
        # plt.scatter(y[0],y[1],label='Seed '+str(seeds_to_test[i]))
        #average_fitness = np.sum(y_arr,axis=1)/3

        plt.plot(avg_arr, label='Pops: '+str(pops_to_test[i]))

    plt.grid()
    plt.ylabel("Average Fitness")
    plt.xlabel("Generations")
    plt.legend()
    plt.show()

if plot_kernel:
    plt.figure()
    for i in range(len(kernel_to_test)):
        data_file = save_directory + 'maco_kernel_' + str(kernel_to_test[i]) + '.dat'
        ancillary_file = save_directory + 'maco_kernel_' + str(kernel_to_test[i]) + '_ancillary.txt'
        file = open(data_file,'rb')
        data = pickle.load(file)[0]
        file.close()
        x = data[0]
        y = data[1]
        y_per_gen = np.array(data[2])
        avg_arr = np.zeros(y_per_gen.shape[0])
        for k in range(y_per_gen.shape[0]):


            gen_avg = 0
            num_valid = 0
            for l in range(y_per_gen.shape[1]):
                avg_one_pop = np.sum(y_per_gen[k,l])/3
                if avg_one_pop < 1000:
                    gen_avg += avg_one_pop
                    num_valid += 1
            avg_arr[k] = gen_avg/num_valid


        time_sim = np.genfromtxt(ancillary_file)
        # plt.scatter(y[0],y[1],label='Seed '+str(seeds_to_test[i]))
        #average_fitness = np.sum(y_arr,axis=1)/3

        plt.plot(avg_arr, label='Kernel: '+str(kernel_to_test[i]))

    plt.grid()
    plt.ylabel("Average Fitness")
    plt.xlabel("Generations")
    plt.legend()
    plt.show()

if plot_convergence_speed:
    plt.figure()
    for i in range(len(convergence_speed_to_test)):
        data_file = save_directory + 'maco_convergence_speed_' + str(convergence_speed_to_test[i]) + '.dat'
        ancillary_file = save_directory + 'maco_convergence_speed_' + str(convergence_speed_to_test[i]) + '_ancillary.txt'
        file = open(data_file,'rb')
        data = pickle.load(file)[0]
        file.close()
        x = data[0]
        y = data[1]
        y_per_gen = np.array(data[2])
        avg_arr = np.zeros(y_per_gen.shape[0])
        for k in range(y_per_gen.shape[0]):


            gen_avg = 0
            num_valid = 0
            for l in range(y_per_gen.shape[1]):
                avg_one_pop = np.sum(y_per_gen[k,l])/3
                if avg_one_pop < 1000:
                    gen_avg += avg_one_pop
                    num_valid += 1
            avg_arr[k] = gen_avg/num_valid


        time_sim = np.genfromtxt(ancillary_file)
        # plt.scatter(y[0],y[1],label='Seed '+str(seeds_to_test[i]))
        #average_fitness = np.sum(y_arr,axis=1)/3

        plt.plot(avg_arr, label='Convergence Speed: '+str(convergence_speed_to_test[i]))

    plt.grid()
    plt.ylabel("Average Fitness")
    plt.xlabel("Generations")
    plt.legend()
    plt.show()

if plot_threshold:
    plt.figure()
    for i in range(len(threshold_to_test)):
        data_file = save_directory + 'maco_threshold_' + str(threshold_to_test[i]) + '.dat'
        ancillary_file = save_directory + 'maco_threshold_' + str(threshold_to_test[i]) + '_ancillary.txt'
        file = open(data_file,'rb')
        data = pickle.load(file)[0]
        file.close()
        x = data[0]
        y = data[1]
        y_per_gen = np.array(data[2])
        avg_arr = np.zeros(y_per_gen.shape[0])
        for k in range(y_per_gen.shape[0]):


            gen_avg = 0
            num_valid = 0
            for l in range(y_per_gen.shape[1]):
                avg_one_pop = np.sum(y_per_gen[k,l])/3
                if avg_one_pop < 1000:
                    gen_avg += avg_one_pop
                    num_valid += 1
            avg_arr[k] = gen_avg/num_valid


        time_sim = np.genfromtxt(ancillary_file)
        # plt.scatter(y[0],y[1],label='Seed '+str(seeds_to_test[i]))
        #average_fitness = np.sum(y_arr,axis=1)/3

        plt.plot(avg_arr, label='Threshold: '+str(threshold_to_test[i]))

    plt.grid()
    plt.ylabel("Average Fitness")
    plt.xlabel("Generations")
    plt.legend()
    plt.show()

if plot_std_convergence_speed:
    plt.figure()
    for i in range(len(std_convergence_speed_to_test)):
        data_file = save_directory + 'maco_std_convergence_speed_' + str(std_convergence_speed_to_test[i]) + '.dat'
        ancillary_file = save_directory + 'maco_std_convergence_speed_' + str(std_convergence_speed_to_test[i]) + '_ancillary.txt'
        file = open(data_file,'rb')
        data = pickle.load(file)[0]
        file.close()
        x = data[0]
        y = data[1]
        y_per_gen = np.array(data[2])
        avg_arr = np.zeros(y_per_gen.shape[0])
        for k in range(y_per_gen.shape[0]):


            gen_avg = 0
            num_valid = 0
            for l in range(y_per_gen.shape[1]):
                avg_one_pop = np.sum(y_per_gen[k,l])/3
                if avg_one_pop < 1000:
                    gen_avg += avg_one_pop
                    num_valid += 1
            avg_arr[k] = gen_avg/num_valid


        time_sim = np.genfromtxt(ancillary_file)
        # plt.scatter(y[0],y[1],label='Seed '+str(seeds_to_test[i]))
        #average_fitness = np.sum(y_arr,axis=1)/3

        plt.plot(avg_arr, label='Std Convergence Speed: '+str(std_convergence_speed_to_test[i]))

    plt.grid()
    plt.ylabel("Average Fitness")
    plt.xlabel("Generations")
    plt.legend()
    plt.show()

