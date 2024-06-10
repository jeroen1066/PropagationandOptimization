import numpy as np
import pickle
import os
import matplotlib.pyplot as plt

data_file = os.getcwd().replace(os.path.basename(os.getcwd()),'') + 'PropagationandOptimization_Jeroen//mcdata.dat'

with open(data_file,'rb') as f:
    b = pickle.load(f)

successful_runs = b[0]

#def plot_R_N(K_vals,A):


for i in range(len(successful_runs)):

    # Unpack the inputs and outputs
    inputs = successful_runs[i][0]
    outputs = successful_runs[i][1]

    R_N = inputs[0]
    R_m = inputs[1]
    L_c = inputs[2]
    theta_c = inputs[3]
    R_s = inputs[4]
    alpha = inputs[5]
    # alpha = np.cos(np.deg2rad(alpha))
    rho = inputs[6]

    # 0 variable terms
    x0 = 1

    # 1 variable terms
    x1 = R_N
    x2 = R_m
    x3 = L_c
    x4 = theta_c
    x5 = R_s  #5  
    x6 = alpha
    x7 = rho


    # 2 variable terms
    # x8 = R_N * R_m # 7
    x9 = R_N * L_c
    x10 = R_N * theta_c
    # x11 = R_N * R_s 10
    x12 = R_N * alpha
    # x13 = R_N * rho

    # x14 = R_m * L_c # 12
    # x15 = R_m * theta_c # 13
    # x16 = R_m * R_s
    x17 = R_m * alpha
    # x18 = R_m * rho

    x19 = L_c * theta_c
    # x20 = L_c * R_s 16
    x21 = L_c * alpha
    # x22 = L_c * rho

    # x23 = theta_c * R_s # 18
    x24 = theta_c * alpha
    # x25 = theta_c * rho

    x26 = R_s * alpha
    # x27 = R_s * rho

    # x28 = alpha * rho

    # 3 variable terms
    # x29 = R_N * R_m * L_c 21
    # x30 = R_N * R_m * theta_c # 22
    # x31 = R_N * R_m * R_s
    x32 = R_N * R_m * alpha
    # x33 = R_N * R_m * rho
    x34 = R_N * L_c * theta_c
    # x35 = R_N * L_c * R_s # 25
    x36 = R_N * L_c * alpha
    # x37 = R_N * L_c * rho
    # x38 = R_N * theta_c * R_s # 27
    x39 = R_N * theta_c * alpha
    # x40 = R_N * theta_c * rho
    x41 = R_N * R_s * alpha
    # x42 = R_N * R_s * rho
    # x43 = R_N * alpha * rho
    x44 = R_m * L_c * theta_c
    # x45 = R_m * L_c * R_s 31
    x46 = R_m * L_c * alpha
    # x47 = R_m * L_c * rho
    # x48 = R_m * theta_c * R_s
    x49 = R_m * theta_c * alpha
    # x50 = R_m * theta_c * rho
    x51 = R_m * R_s * alpha
    # x52 = R_m * R_s * rho
    # x53 = R_m * alpha * rho

    x54 = L_c * theta_c * R_s
    x55 = L_c * theta_c * alpha
    # x56 = L_c * theta_c * rho
    x57 = L_c * R_s * alpha
    # x58 = L_c * R_s * rho
    # x59 = L_c * alpha * rho

    x60 = theta_c * R_s * alpha
    # x61 = theta_c * R_s * rho
    # x62 = theta_c * alpha * rho

    # x63 = R_s * alpha * rho

    # 4 variable terms
    # x64 = R_N * R_m * L_c * theta_c # 39
    # x65 = R_N * R_m * L_c * R_s
    x66 = R_N * R_m * L_c * alpha
    # x67 = R_N * R_m * L_c * rho
    # x68 = R_N * R_m * theta_c * R_s
    x69 = R_N * R_m * theta_c * alpha
    # x70 = R_N * R_m * theta_c * rho
    x71 = R_N * R_m * R_s * alpha
    # x72 = R_N * R_m * R_s * rho
    # x73 = R_N * R_m * alpha * rho
    # x74 = R_N * L_c * theta_c * R_s # 43
    x75 = R_N * L_c * theta_c * alpha
    # x76 = R_N * L_c * theta_c * rho
    # x77 = R_N * L_c * R_s * alpha # 45
    # x78 = R_N * L_c * R_s * rho
    # x79 = R_N * L_c * alpha * rho
    x80 = R_N * theta_c * R_s * alpha
    # x81 = R_N * theta_c * R_s * rho
    # x82 = R_N * theta_c * alpha * rho
    # x83 = R_N * R_s * alpha * rho

    # x84 = R_m * L_c * theta_c * R_s # 47
    x85 = R_m * L_c * theta_c * alpha
    # x86 = R_m * L_c * theta_c * rho
    x87 = R_m * L_c * R_s * alpha
    # x88 = R_m * L_c * R_s * rho
    # x89 = R_m * L_c * alpha * rho
    x90 = R_m * theta_c * R_s * alpha
    # x91 = R_m * theta_c * R_s * rho
    # x92 = R_m * theta_c * alpha * rho
    # x93 = R_m * R_s * alpha * rho

    x94 = L_c * theta_c * R_s * alpha
    # x95 = L_c * theta_c * R_s * rho
    # x96 = L_c * theta_c * alpha * rho
    # x97 = L_c * R_s * alpha * rho

    # x98 = theta_c * R_s * alpha * theta_c

    # 5 variable terms
    # x99 = R_N * R_m * L_c * theta_c * R_s #52
    x100 = R_N * R_m * L_c * theta_c * alpha
    # x101 = R_N * R_m * L_c * theta_c * rho
    # x102 = R_N * R_m * L_c * R_s * alpha
    # x103 = R_N * R_m * L_c * R_s * rho
    # x104 = R_N * R_m * L_c * alpha * rho
    x105 = R_N * R_m * theta_c * R_s * alpha
    # x106 = R_N * R_m * theta_c * R_s * rho
    # x107 = R_N * R_m * theta_c * alpha * rho
    # x108 = R_N * R_m * R_s * alpha * rho
    x109 = R_N * L_c * theta_c * R_s * alpha
    # x110 = R_N * L_c * theta_c * R_s * rho
    # x111 = R_N * L_c * theta_c * alpha * rho
    # x112 = R_N * L_c * R_s * alpha * rho
    # x113 = R_N * theta_c * R_s * alpha * rho

    x114 = R_m * L_c * theta_c * R_s * alpha
    # x115 = R_m * L_c * theta_c * R_s * rho
    # x116 = R_m * L_c * theta_c * alpha * rho
    # x117 = R_m * L_c * R_s * alpha * rho
    # x118 = R_m * theta_c * R_s * alpha * rho

    # x119 = L_c * theta_c * R_s * alpha * rho

    # 6 variable terms
    # x120 = R_N * R_m * L_c * theta_c * R_s * alpha
    # x121 = R_N * R_m * L_c * theta_c * R_s * rho
    # x122 = R_N * R_m * L_c * theta_c * alpha * rho
    # x123 = R_N * R_m * L_c * R_s * alpha * rho
    # x124 = R_N * R_m * theta_c * R_s * alpha * rho
    # x125 = R_N * L_c * theta_c * R_s * alpha * rho
    # x126 = R_m * L_c * theta_c * R_s * alpha * rho

    # 7 variable term
    # x127 = R_N * R_m * L_c * theta_c * R_s * alpha * rho

    # outputs
    volume_i = outputs[0]
    LD_i = outputs[1]
    Gload_i = outputs[2]

    # set up least squares regression vector
    A_i = np.asarray([ x0, x1, x2, x3, x4, x6, x9, x10, x12, x17, x19, x21, 
                      x24, x26, x32, x34,  x36,  x39, x41, x44,  x46, x49, x51, x54, 
                      x55, x57, x60,  x66, x69, x71,  x75, x80,  x85, x87, x90, x94, 
                        x100, x105, x109, x114])
    
    #   A_i = np.asarray([ x0, x1, x2, x3, x4, x5, x6, x8, x9, x10, x11, x12, x14, x15, x17, x19, x20, x21, x23, 
    #                   x24, x26, x29, x30, x32, x34, x35, x36, x38, x39, x41, x44, x45, x46, x49, x51, x54, 
    #                   x55, x57, x60, x64, x66, x69, x71, x74, x75, x77, x80, x84, x85, x87, x90, x94, x99,
    #                     x100, x105, x109, x114])

    # add vector and outputs to matrices

    if i == 0:
        A = A_i
        volume = volume_i
        LD = LD_i
        Gload = Gload_i
    else:
        A = np.vstack((A,A_i))
        volume = np.vstack((volume,volume_i))
        LD = np.vstack((LD,LD_i))
        Gload = np.vstack((Gload,Gload_i))


# Gload
# [ K0, K1, K2, K3, K4, K5, K6, K8, K9, K10, K11, K12, 
#  K14, K15, K17, K19, K20, K21, K23, K24, K26, K29, K30, 
#  K32, K34, K35, K36, K38, K39, K41, K44, K45, K46, K49, 
#  K51, K54, K55, K57, K60, K64, K66, K69, K71, K74, K75, 
#  K77, K80, K84, K85, K87, K90, K94, K99, K100, K105, K109, K114 ]= np.linalg.lstsq(A, Gload, rcond=None)[0]

[K0, K1, K2, K3, K4, K6, K9, K10, K12, K17, K19, K21, 
 K24, K26, K32, K34, K36, K39, K41, K44, K46, K49, K51, 
 K54, K55, K57, K60, K66, K69, K71, K75, K80, K85, K87, 
 K90, K94, K100, K105, K109, K114] = np.linalg.lstsq(A, Gload, rcond=None)[0]

Gload_RSS = np.linalg.lstsq(A, Gload, rcond=None)[1]

K_vals = [K0, K1, K2, K3, K4, K6, K9, K10, K12, K17, K19, K21, 
 K24, K26, K32, K34, K36, K39, K41, K44, K46, K49, K51, 
 K54, K55, K57, K60, K66, K69, K71, K75, K80, K85, K87, 
 K90, K94, K100, K105, K109, K114] 

# analysis
Gload_sim = []
Gload_mod = []
Gload_res = []

R_N_list = []
R_m_list = []
L_c_list = []
theta_c_list = []
R_s_list = []
alpha_list = []
rho_list = []

for i in range(len(A)):
    A_i = A[i]
    aeroG_i = Gload[i]

    R_N_i = A_i[1]
    R_m_i = A_i[2]
    L_c_i = A_i[3]
    theta_c_i = A_i[4]
    R_s_i = A_i[5]
    alpha_i = A_i[6]
    rho_i = A_i[7]

    Gload_mod_i = 0
    for j in range(len(A_i)):
        Gload_mod_j = A_i[j] * K_vals[j]
        Gload_mod_i = Gload_mod_i + Gload_mod_j

    Gload_res_i = Gload_mod_i - aeroG_i

    Gload_sim.append(aeroG_i)
    Gload_mod.append(Gload_mod_i)
    Gload_res.append(Gload_res_i)

    R_N_list.append(R_N_i)
    R_m_list.append(R_m_i)
    L_c_list.append(L_c_i)
    theta_c_list.append(theta_c_i)
    R_s_list.append(R_s_i)
    alpha_list.append(alpha_i)
    rho_list.append(rho_i)

mean_Gload = sum(Gload_sim)/len(Gload_sim)
Gload_TSS = 0
for i in range(len(Gload_sim)):
    TSS_i = (Gload_sim[i] - mean_Gload) ** 2
    Gload_TSS = Gload_TSS + TSS_i

R2_Gload = 1 - (Gload_RSS[0]/Gload_TSS[0])
print('R^2 value Gload: ', R2_Gload)
print('R value Gload:', np.sqrt(R2_Gload))

label1 = 'Simulated Data'
label2 = 'Response Surface'
label3 = 'Residuals'

fig = plt.figure(figsize=(8, 8))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
ax1.scatter(R_N_list,Gload_sim,color='blue',label=label1)
ax1.scatter(R_N_list,Gload_mod,color='red',label=label2)
ax1.set_xlabel('R_N [m]', size=16)
ax1.set_ylabel('G-load [-]', size=16)
ax1.legend(fontsize=16)
ax1.grid()
ax2.scatter(R_N_list,Gload_res,color='black',label=label3)
ax2.set_xlabel('R_N [m]', size=16)
ax2.set_ylabel('G-load [-]', size=16)
ax2.legend(fontsize=16)
ax2.grid()
plt.show()

fig = plt.figure(figsize=(8, 8))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
ax1.scatter(R_m_list,Gload_sim,color='blue',label=label1)
ax1.scatter(R_m_list,Gload_mod,color='red',label=label2)
ax1.set_xlabel('R_m [m]', size=16)
ax1.set_ylabel('G-load [-]', size=16)
ax1.legend(fontsize=16)
ax1.grid()
ax2.scatter(R_m_list,Gload_res,color='black',label=label3)
ax2.set_xlabel('R_m [m]', size=16)
ax2.set_ylabel('G-load [-]', size=16)
ax2.legend(fontsize=16)
ax2.grid()
plt.show()

fig = plt.figure(figsize=(8, 8))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
ax1.scatter(L_c_list,Gload_sim,color='blue',label=label1)
ax1.scatter(L_c_list,Gload_mod,color='red',label=label2)
ax1.set_xlabel('L_c [m]', size=16)
ax1.set_ylabel('G-load [-]', size=16)
ax1.legend(fontsize=16)
ax1.grid()
ax2.scatter(L_c_list,Gload_res,color='black',label=label3)
ax2.set_xlabel('L_c [m]', size=16)
ax2.set_ylabel('G-load [-]', size=16)
ax2.legend(fontsize=16)
ax2.grid()
plt.show()

fig = plt.figure(figsize=(8, 8))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
ax1.scatter(theta_c_list,Gload_sim,color='blue',label=label1)
ax1.scatter(theta_c_list,Gload_mod,color='red',label=label2)
ax1.set_xlabel('θ_c [rad]', size=16)
ax1.set_ylabel('G-load [-]', size=16)
ax1.legend(fontsize=16)
ax1.grid()
ax2.scatter(theta_c_list,Gload_res,color='black',label=label3)
ax2.set_xlabel('θ_c [rad]', size=16)
ax2.set_ylabel('G-load [-]', size=16)
ax2.legend(fontsize=16)
ax2.grid()
plt.show()

fig = plt.figure(figsize=(8, 8))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
ax1.scatter(R_s_list,Gload_sim,color='blue',label=label1)
ax1.scatter(R_s_list,Gload_mod,color='red',label=label2)
ax1.set_xlabel('R_s [m]', size=16)
ax1.set_ylabel('G-load [-]', size=16)
ax1.legend(fontsize=16)
ax1.grid()
ax2.scatter(R_s_list,Gload_res,color='black',label=label3)
ax2.set_xlabel('R_s [m]', size=16)
ax2.set_ylabel('G-load [-]', size=16)
ax2.legend(fontsize=16)
ax2.grid()
plt.show()

fig = plt.figure(figsize=(8, 8))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
ax1.scatter(alpha_list,Gload_sim,color='blue',label=label1)
ax1.scatter(alpha_list,Gload_mod,color='red',label=label2)
ax1.set_xlabel('α [rad]', size=16)
ax1.set_ylabel('G-load [-]', size=16)
ax1.legend(fontsize=16)
ax1.grid()
ax2.scatter(alpha_list,Gload_res,color='black',label=label3)
ax2.set_xlabel('α [rad]', size=16)
ax2.set_ylabel('G-load [-]', size=16)
ax2.legend(fontsize=16)
ax2.grid()
plt.show()

fig = plt.figure(figsize=(8, 8))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
ax1.scatter(rho_list,Gload_sim,color='blue',label=label1)
ax1.scatter(rho_list,Gload_mod,color='red',label=label2)
ax1.set_xlabel('ρ [kg/m^3]', size=16)
ax1.set_ylabel('G-load [-]', size=16)
ax1.legend(fontsize=16)
ax1.grid()
ax2.scatter(rho_list,Gload_res,color='black',label=label3)
ax2.set_xlabel('ρ [kg/m^3]', size=16)
ax2.set_ylabel('G-load [-]', size=16)
ax2.legend(fontsize=16)
ax2.grid()
plt.show()


# plt.scatter(np.arange(0,len(K_vals),1),K_vals,label='Response Surface Coefficients for G-load')
minor_impact_coeff = []
major_impact_coeff = []
impact_threshold = 1
for i, val in enumerate(K_vals):
    val = np.abs(val)
    if val < impact_threshold:
        plt.scatter(i, val, color='red')
        minor_impact_coeff.append(i)
    if val > impact_threshold:
        plt.scatter(i, val, color='blue')

    if val > 70:
        major_impact_coeff.append(i)
        plt.scatter(i, val, color='green')

plt.xlabel('coefficient term [-]', size=16)
plt.ylabel('coefficient value [-]', size=16)
plt.yscale('log')
plt.title('Response Surface Coefficients for G-load', size=16)
# plt.legend(fontsize=16)
plt.grid()

# Add code to highlight values below 1 in red


plt.show()

# plt.bar(np.arange(0, len(K_vals)), K_vals)
# plt.xlabel('Coefficient term [-]', size=16)
# plt.ylabel('Coefficient value [-]', size=16)
# # plt.yscale('log')
# plt.title('Response Surface Coefficients for G-load', size=16)
# plt.grid()
# plt.show()


