import numpy as np

with open('mcdata','rb') as f:
    b = pickle.load(f)

successful_runs = b[0]

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
    x1 = 1

    # 1 variable terms
    x2 = R_N
    x3 = R_m
    x4 = L_c
    x5 = theta_c
    x6 = R_s
    x7 = alpha
    x8 = rho

    # 2 variable terms
    x9 = R_N * R_m
    x10 = R_N * L_c
    x11 = R_N * theta_c
    x12 = R_N * R_s
    x13 = R_N * alpha
    x14 = R_N * rho

    x15 = R_m * L_c
    x16 = R_m * theta_c
    x17 = R_m * R_s
    x18 = R_m * alpha
    x19 = R_m * rho

    x20 = L_c * theta_c
    x21 = L_c * R_s
    x22 = L_c * alpha
    x23 = L_c * rho

    x24 = theta_c * R_s
    x25 = theta_c * alpha
    x26 = theta_c * rho

    x27 = R_s * alpha
    x28 = R_s * rho

    x29 = alpha * rho

    # 3 variable terms
    x30 = R_N * R_m * L_c
    x31 = R_N * R_m * theta_c
    x32 = R_N * R_m * R_s
    x33 = R_N * R_m * alpha
    x34 = R_N * R_m * rho
    x35 = R_N * L_c * theta_c
    x36 = R_N * L_c * R_s
    x37 = R_N * L_c * alpha
    x38 = R_N * L_c * rho
    x39 = R_N * theta_c * R_s
    x40 = R_N * theta_c * alpha
    x41 = R_N * theta_c * rho
    x42 = R_N * R_s * alpha
    x43 = R_N * R_s * rho
    x44 = R_N * alpha * rho

    x45 = R_m * L_c * theta_c
    x46 = R_m * L_c * R_s
    x47 = R_m * L_c * alpha
    x48 = R_m * L_c * rho
    x49 = R_m * theta_c * R_s
    x50 = R_m * theta_c * alpha
    x51 = R_m * theta_c * rho
    x52 = R_m * R_s * alpha
    x53 = R_m * R_s * rho
    x54 = R_m * alpha * rho

    x55 = L_c * theta_c * R_s
    x56 = L_c * theta_c * alpha
    x57 = L_c * theta_c * rho
    x58 = L_c * R_s * alpha
    x59 = L_c * R_s * rho
    x60 = L_c * alpha * rho

    x61 = theta_c * R_s * alpha
    x62 = theta_c * R_s * rho
    x63 = theta_c * alpha * rho

    x64 = R_s * alpha * rho

    # 4 variable terms
    x65 = R_N * R_m * L_c * theta_c
    x66 = R_N * R_m * L_c * R_s
    x67 = R_N * R_m * L_c * alpha
    x68 = R_N * R_m * L_c * rho
    x69 = R_N * R_m * theta_c * R_s
    x70 = R_N * R_m * theta_c * alpha
    x71 = R_N * R_m * theta_c * rho
    x72 = R_N * R_m * R_s * alpha
    x73 = R_N * R_m * R_s * rho
    x74 = R_N * R_m * alpha * rho
    x75 = R_N * L_c * theta_c * R_s
    x76 = R_N * L_c * theta_c * alpha
    x77 = R_N * L_c * theta_c * rho
    x78 = R_N * L_c * R_s * alpha
    x79 = R_N * L_c * R_s * rho
    x80 = R_N * L_c * alpha * rho
    x81 = R_N * theta_c * R_s * alpha
    x82 = R_N * theta_c * R_s * rho
    x83 = R_N * theta_c * alpha * rho
    x84 = R_N * R_s * alpha * rho

    x85 = R_m * L_c * theta_c * R_s
    x86 = R_m * L_c * theta_c * alpha
    x87 = R_m * L_c * theta_c * rho
    x88 = R_m * L_c * R_s * alpha
    x89 = R_m * L_c * R_s * rho
    x91 = R_m * L_c * alpha * rho
    x92 = R_m * theta_c * R_s * alpha
    x93 = R_m * theta_c * R_s * rho
    x94 = R_m * theta_c * alpha * rho
    x95 = R_m * R_s * alpha * rho

    x96 = L_c * theta_c * R_s * alpha
    x97 = L_c * theta_c * R_s * rho
    x98 = L_c * theta_c * alpha * rho
    x99 = L_c * R_s * alpha * rho

    x100 = theta_c * R_s * alpha * theta_c

    # 5 variable terms
    x101 = R_N * R_m * L_c * theta_c * R_s
    x102 = R_N * R_m * L_c * theta_c * alpha
    x103 = R_N * R_m * L_c * theta_c * rho
    x104 = R_N * R_m * L_c * R_s * alpha
    x105 = R_N * R_m * L_c * R_s * rho
    x106 = R_N * R_m * L_c * alpha * rho
    x107 = R_N * R_m * theta_c * R_s * alpha
    x108 = R_N * R_m * theta_c * R_s * rho
    x109 = R_N * R_m * theta_c * alpha * rho
    x110 = R_N * R_m * R_s * alpha * rho
    x111 = R_N * L_c * theta_c * R_s * alpha
    x112 = R_N * L_c * theta_c * R_s * rho
    x113 = R_N * L_c * theta_c * alpha * rho
    x114 = R_N * L_c * R_s * alpha * rho
    x115 = R_N * theta_c * R_s * alpha * rho

    x116 = R_m * L_c * theta_c * R_s * alpha
    x177 = R_m * L_c * theta_c * R_s * rho
    x118 = R_m * L_c * theta_c * alpha * rho
    x119 = R_m * L_c * R_s * alpha * rho
    x120 = R_m * theta_c * R_s * alpha * rho

    x121 = L_c * theta_c * R_s * alpha * rho

    # 6 variable terms
    x122 = R_N * R_m * L_c * theta_c * R_s * alpha
    x123 = R_N * R_m * L_c * theta_c * R_s * rho
    x124 = R_N * R_m * L_c * theta_c * alpha * rho
    x125 = R_N * R_m * L_c * R_s * alpha * rho
    x126 = R_N * R_m * theta_c * R_s * alpha * rho
    x127 = R_N * L_c * theta_c * R_s * alpha * rho
    x128 = R_m * L_c * theta_c * R_s * alpha * rho

    # 7 variable term
    x129 = R_N * R_m * L_c * theta_c * R_s * alpha * rho

    # outputs
    volume_i = outputs[0]
    LD_i = outputs[1]
    Gload_i = outputs[2]

    # set up least squares regression vector
    A_i = np.asarray([x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,x16,x17,x18,x19,x20,x21,x22,x23,x24,x25,x26,x27,x28,x29,
           x30,x31,x32,x33,x34,x35,x36,x37,x38,x39,x40,x41,x42,x43,x44,x45,x46,x47,x48,x49,x50,x51,x52,x53,x54,x55,x56,
           x57,x58,x59,x60,x61,x62,x63,x64,x65,x66,x67,x68,x69,x70,x71,x72,x73,x74,x75,x76,x77,x78,x79,x80,x81,x82,x83,
           x84,x85,x86,x87,x88,x89,x90,x91,x92,x93,x94,x95,x96,x97,x98,x99,x100,x101,x102,x103,x104,x105,x106,x107,x108,
           x109,x110,x111,x112,x113,x114,x115,x116,x117,x118,x119,x120,x121,x122,x123,x124,x125,x126,x127,x128,x129])

    # add vector and outputs to matrices

    if i == 0:
        A = A_i
        volume = volume_i
        LD = LD_i
        Gload = Gload_i
    else:
        np.vstack((A,A_i))
        np.vstack((volume,volume_i))
        np.vstack((LD,LD_i))
        np.vstack((Gload,Gload_i))

K1,K2,K3,K4,K5,K6,K7,K8,K9,K10,K11,K12,K13,K14,K15,K16,K17,K18,K19,K20,K21,K22,K23,K24,K25,K26,K27,K28,K29,K30,K31,K32,\
K33,K34,K35,K36,K37,K38,K39,K40,K41,K42,K43,K44,K45,K46,K47,K48,K49,K50,K51,K52,K53,K54,K55,K56,K57,K58,K59,K60,K61,\
K62,K63,K64,K65,K66,K67,K68,K69,K70,K71,K72,K73,K74,K75,K76,K77,K78,K79,K80,K81,K82,K83,K84,K85,K86,K87,K88,K89,K90,\
K91,K92,K93,K94,K95,K96,K97,K98,K99,K100,K101,K102,K103,K104,K105,K106,K107,K108,K109,K110,K111,K112,K113,K114,K115,\
K116,K117,K118,K119,K120,K121,K122,K123,K124,K125,K126,K127,K128,K129 = np.linalg.lstsq(A, volume, rcond=None)[0]



