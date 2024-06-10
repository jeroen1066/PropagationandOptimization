import numpy as np
import matplotlib.pyplot as plt
from CapsuleEntryPropagation import number_of_tolerances, current_dir


tolerances_test = [ 10.0 ** (-14.0 + test_tolerance) for test_tolerance in range(number_of_tolerances)]
tolerance_labels = [f'Tol = {tolerance}' for tolerance in tolerances_test]
# tolerance_labels = tolerance_labels[1:]
tolerances_test = tolerances_test[1:]

plot1 = True

if plot1:
    #plot runtime, getting data from ancillary file
    # plt.figure()
    times_required_array = []
    max_difference_array = []
    for tolerance_no in range(number_of_tolerances-1):
        tolerance_no = tolerance_no + 1

        print(f'Plot2: Plotting model {tolerance_no}: {tolerance_labels[tolerance_no-1]}')

        data = np.genfromtxt(f'{current_dir}/Tolerance_{tolerance_no}/state_difference_wrt_nominal_case.dat')
        difference_norm = np.linalg.norm(data[:, 1:3], axis=1)
        max_difference = max(difference_norm)
        max_difference_array.append(max_difference)

        function_data = np.genfromtxt(f'{current_dir}/Tolerance_{tolerance_no}/ancillary_simulation_info.txt', dtype = str, delimiter = '\t')
        time_required = float(function_data[3][1])
        times_required_array.append(time_required)
       
    fig, ax1 = plt.subplots()

    ax1.plot(tolerances_test, times_required_array, color='blue')
    ax1.set_xlabel('Tolerance')
    ax1.set_xscale('log')
    ax1.set_ylabel('Runtime [s]', color='blue')  # Set y-axis label color to blue
    # ax1.set_xticklabels(tolerance_labels, rotation=90)
    # ax1.grid()

    ax2 = ax1.twinx()
    ax2.plot(tolerances_test, max_difference_array, color='red')
    ax2.set_ylabel('Max Difference [m]', color='red')  # Set y-axis label color to red
    # ax2.grid()
    ax2.set_yscale('log')


    plt.tight_layout()
    plt.show()
