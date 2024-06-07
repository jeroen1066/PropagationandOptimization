import numpy as np
import pickle
import os
import matplotlib.pyplot as plt

data_file = os.getcwd().replace(os.path.basename(os.getcwd()),'') + 'mcdata.dat'

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
    rho = inputs[6]

    # 0 variable terms
    x0 = 1

    # 1 variable terms
    x1 = R_N
    x2 = R_m
    x3 = L_c
    x4 = theta_c
    x5 = R_s
    x6 = alpha
    x7 = rho

    # 2 variable terms
    x8 = R_N * R_m
    x9 = R_N * L_c
    x10 = R_N * theta_c
    x11 = R_N * R_s
    x12 = R_N * alpha
    x13 = R_N * rho

    x14 = R_m * L_c
    x15 = R_m * theta_c
    x16 = R_m * R_s
    x17 = R_m * alpha
    x18 = R_m * rho

    x19 = L_c * theta_c
    x20 = L_c * R_s
    x21 = L_c * alpha
    x22 = L_c * rho

    x23 = theta_c * R_s
    x24 = theta_c * alpha
    x25 = theta_c * rho

    x26 = R_s * alpha
    x27 = R_s * rho

    x28 = alpha * rho

    # 3 variable terms
    x29 = R_N * R_m * L_c
    x30 = R_N * R_m * theta_c
    x31 = R_N * R_m * R_s
    x32 = R_N * R_m * alpha
    x33 = R_N * R_m * rho
    x34 = R_N * L_c * theta_c
    x35 = R_N * L_c * R_s
    x36 = R_N * L_c * alpha
    x37 = R_N * L_c * rho
    x38 = R_N * theta_c * R_s
    x39 = R_N * theta_c * alpha
    x40 = R_N * theta_c * rho
    x41 = R_N * R_s * alpha
    x42 = R_N * R_s * rho
    x43 = R_N * alpha * rho
    x44 = R_m * L_c * theta_c
    x45 = R_m * L_c * R_s
    x46 = R_m * L_c * alpha
    x47 = R_m * L_c * rho
    x48 = R_m * theta_c * R_s
    x49 = R_m * theta_c * alpha
    x50 = R_m * theta_c * rho
    x51 = R_m * R_s * alpha
    x52 = R_m * R_s * rho
    x53 = R_m * alpha * rho

    x54 = L_c * theta_c * R_s
    x55 = L_c * theta_c * alpha
    x56 = L_c * theta_c * rho
    x57 = L_c * R_s * alpha
    x58 = L_c * R_s * rho
    x59 = L_c * alpha * rho

    x60 = theta_c * R_s * alpha
    x61 = theta_c * R_s * rho
    x62 = theta_c * alpha * rho

    x63 = R_s * alpha * rho

    # 4 variable terms
    x64 = R_N * R_m * L_c * theta_c
    x65 = R_N * R_m * L_c * R_s
    x66 = R_N * R_m * L_c * alpha
    x67 = R_N * R_m * L_c * rho
    x68 = R_N * R_m * theta_c * R_s
    x69 = R_N * R_m * theta_c * alpha
    x70 = R_N * R_m * theta_c * rho
    x71 = R_N * R_m * R_s * alpha
    x72 = R_N * R_m * R_s * rho
    x73 = R_N * R_m * alpha * rho
    x74 = R_N * L_c * theta_c * R_s
    x75 = R_N * L_c * theta_c * alpha
    x76 = R_N * L_c * theta_c * rho
    x77 = R_N * L_c * R_s * alpha
    x78 = R_N * L_c * R_s * rho
    x79 = R_N * L_c * alpha * rho
    x80 = R_N * theta_c * R_s * alpha
    x81 = R_N * theta_c * R_s * rho
    x82 = R_N * theta_c * alpha * rho
    x83 = R_N * R_s * alpha * rho

    x84 = R_m * L_c * theta_c * R_s
    x85 = R_m * L_c * theta_c * alpha
    x86 = R_m * L_c * theta_c * rho
    x87 = R_m * L_c * R_s * alpha
    x88 = R_m * L_c * R_s * rho
    x89 = R_m * L_c * alpha * rho
    x90 = R_m * theta_c * R_s * alpha
    x91 = R_m * theta_c * R_s * rho
    x92 = R_m * theta_c * alpha * rho
    x93 = R_m * R_s * alpha * rho

    x94 = L_c * theta_c * R_s * alpha
    x95 = L_c * theta_c * R_s * rho
    x96 = L_c * theta_c * alpha * rho
    x97 = L_c * R_s * alpha * rho

    x98 = theta_c * R_s * alpha * theta_c

    # 5 variable terms
    x99 = R_N * R_m * L_c * theta_c * R_s
    x100 = R_N * R_m * L_c * theta_c * alpha
    x101 = R_N * R_m * L_c * theta_c * rho
    x102 = R_N * R_m * L_c * R_s * alpha
    x103 = R_N * R_m * L_c * R_s * rho
    x104 = R_N * R_m * L_c * alpha * rho
    x105 = R_N * R_m * theta_c * R_s * alpha
    x106 = R_N * R_m * theta_c * R_s * rho
    x107 = R_N * R_m * theta_c * alpha * rho
    x108 = R_N * R_m * R_s * alpha * rho
    x109 = R_N * L_c * theta_c * R_s * alpha
    x110 = R_N * L_c * theta_c * R_s * rho
    x111 = R_N * L_c * theta_c * alpha * rho
    x112 = R_N * L_c * R_s * alpha * rho
    x113 = R_N * theta_c * R_s * alpha * rho

    x114 = R_m * L_c * theta_c * R_s * alpha
    x115 = R_m * L_c * theta_c * R_s * rho
    x116 = R_m * L_c * theta_c * alpha * rho
    x117 = R_m * L_c * R_s * alpha * rho
    x118 = R_m * theta_c * R_s * alpha * rho

    x119 = L_c * theta_c * R_s * alpha * rho

    # 6 variable terms
    x120 = R_N * R_m * L_c * theta_c * R_s * alpha
    x121 = R_N * R_m * L_c * theta_c * R_s * rho
    x122 = R_N * R_m * L_c * theta_c * alpha * rho
    x123 = R_N * R_m * L_c * R_s * alpha * rho
    x124 = R_N * R_m * theta_c * R_s * alpha * rho
    x125 = R_N * L_c * theta_c * R_s * alpha * rho
    x126 = R_m * L_c * theta_c * R_s * alpha * rho

    # 7 variable term
    x127 = R_N * R_m * L_c * theta_c * R_s * alpha * rho

    # outputs
    volume_i = outputs[0]
    LD_i = outputs[1]
    Gload_i = outputs[2]

    # set up least squares regression vector
    A_i = np.asarray([x0,x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,x16,x17,x18,x19,x20,x21,x22,x23,x24,x25,x26,
                      x27,x28,x29,x30,x31,x32,x33,x34,x35,x36,x37,x38,x39,x40,x41,x42,x43,x44,x45,x46,x47,x48,x49,x50,
                      x51,x52,x53,x54,x55,x56,x57,x58,x59,x60,x61,x62,x63,x64,x65,x66,x67,x68,x69,x70,x71,x72,x73,x74,
                      x75,x76,x77,x78,x79,x80,x81,x82,x83,x84,x85,x86,x87,x88,x89,x90,x91,x92,x93,x94,x95,x96,x97,x98,
                      x99,x100,x101,x102,x103,x104,x105,x106,x107,x108,x109,x110,x111,x112,x113,x114,x115,x116,x117,
                      x118,x119,x120,x121,x122,x123,x124,x125,x126,x127])

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

K0,K1,K2,K3,K4,K5,K6,K7,K8,K9,K10,K11,K12,K13,K14,K15,K16,K17,K18,K19,K20,K21,K22,K23,K24,K25,K26,K27,K28,K29,K30,K31,\
K32,K33,K34,K35,K36,K37,K38,K39,K40,K41,K42,K43,K44,K45,K46,K47,K48,K49,K50,K51,K52,K53,K54,K55,K56,K57,K58,K59,K60,\
K61,K62,K63,K64,K65,K66,K67,K68,K69,K70,K71,K72,K73,K74,K75,K76,K77,K78,K79,K80,K81,K82,K83,K84,K85,K86,K87,K88,K89,\
K90,K91,K92,K93,K94,K95,K96,K97,K98,K99,K100,K101,K102,K103,K104,K105,K106,K107,K108,K109,K110,K111,K112,K113,K114,\
K115,K116,K117,K118,K119,K120,K121,K122,K123,K124,K125,K126,K127 = np.linalg.lstsq(A, volume, rcond=None)[0]

K_vals = [K0,K1,K2,K3,K4,K5,K6,K7,K8,K9,K10,K11,K12,K13,K14,K15,K16,K17,K18,K19,K20,K21,K22,K23,K24,K25,K26,K27,K28,K29,K30,K31,\
K32,K33,K34,K35,K36,K37,K38,K39,K40,K41,K42,K43,K44,K45,K46,K47,K48,K49,K50,K51,K52,K53,K54,K55,K56,K57,K58,K59,K60,\
K61,K62,K63,K64,K65,K66,K67,K68,K69,K70,K71,K72,K73,K74,K75,K76,K77,K78,K79,K80,K81,K82,K83,K84,K85,K86,K87,K88,K89,\
K90,K91,K92,K93,K94,K95,K96,K97,K98,K99,K100,K101,K102,K103,K104,K105,K106,K107,K108,K109,K110,K111,K112,K113,K114,\
K115,K116,K117,K118,K119,K120,K121,K122,K123,K124,K125,K126,K127]

# analysis

'''
# 1 variable terms
for i in range(1,8):
    K_i = K_vals[i]

    x_vals = []
    V_response = []
    V = []
    for j in range(len(A)):
        x = A[j][i]
        x_vals.append(x)

        V_response_j = x * K_i
        V_response.append(V_response_j)

        V_j = volume[j][0]
        V.append(V_j)

    plt.scatter(x_vals,V)
    plt.scatter(x_vals,V_response)
    plt.show()
'''

Vol_sim = []
Vol_mod = []

R_N_list = []
R_m_list = []
L_c_list = []
theta_c_list = []
R_s_list = []
alpha_list = []
rho_list = []

for i in range(len(A)):
    A_i = A[i]
    Vol_i = volume[i]

    R_N_i = A_i[1]
    R_m_i = A_i[2]
    L_c_i = A_i[3]
    theta_c_i = A_i[4]
    R_s_i = A_i[5]
    alpha_i = A_i[6]
    rho_i = A_i[7]

    Vol_mod_i = 0
    for j in range(len(A_i)):
        Vol_mod_j = A_i[j] * K_vals[j]
        Vol_mod_i = Vol_mod_i + Vol_mod_j

    Vol_sim.append(Vol_i)
    Vol_mod.append(Vol_mod_i)

    R_N_list.append(R_N_i)
    R_m_list.append(R_m_i)
    L_c_list.append(L_c_i)
    theta_c_list.append(theta_c_i)
    R_s_list.append(R_s_i)
    alpha_list.append(alpha_i)
    rho_list.append(rho_i)

plt.scatter(R_N_list,Vol_sim)
plt.scatter(R_N_list,Vol_mod)
plt.show()

plt.scatter(R_m_list,Vol_sim)
plt.scatter(R_m_list,Vol_mod)
plt.show()

plt.scatter(L_c_list,Vol_sim)
plt.scatter(L_c_list,Vol_mod)
plt.show()

plt.scatter(theta_c_list,Vol_sim)
plt.scatter(theta_c_list,Vol_mod)
plt.show()

plt.scatter(R_s_list,Vol_sim)
plt.scatter(R_s_list,Vol_mod)
plt.show()

plt.scatter(alpha_list,Vol_sim)
plt.scatter(alpha_list,Vol_mod)
plt.show()

plt.scatter(rho_list,Vol_sim)
plt.scatter(rho_list,Vol_mod)
plt.show()







