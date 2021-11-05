# -*- coding: utf-8 -*-
"""
Created on Sun Oct 31 01:09:36 2021

@author: joeja
"""

import matplotlib.pyplot as pp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import torch.optim as optim
from scipy.ndimage import median_filter

# build dataset
df = pd.read_csv('https://health-infobase.canada.ca/src/data/covidLive/covid19-download.csv')
T = pd.to_datetime(df.date)
first = T.min()
t = T.map(lambda date: (pd.Timestamp(date) - first).days)
df['t'] = t.to_numpy()

provlist = ["Alberta", "British Columbia", "Manitoba", "New Brunswick", "Newfoundland and Labrador",
            "Northwest Territories",
            "Nova Scotia", "Nunavut", "Ontario", "Prince Edward Island", "Quebec", "Saskatchewan", "Yukon"]
poplist = [4442879, 5214805, 1383765, 789225, 520553, 45504, 992055, 39403, 14826276, 164318, 8604495, 1179844,
           42986]  # according to wikipedia
provdatalist = list()
for i in range(len(provlist)):
    clean_df = pd.DataFrame(columns=['t'], index=range(638))
    clean_df.t = pd.Series(df.index)
    A = df.loc[df['prname'] == provlist[i]]
    A = A[["t", "numactive"]]
    A["numactive"] = median_filter(A["numactive"], 7, mode='mirror') / poplist[i]
    clean_df = pd.merge(clean_df, A, how="left", on="t")
    clean_df.numactive = clean_df.numactive.replace(np.nan, 0)
    provdatalist.append(clean_df)

n = len(provdatalist[0].numactive) - 1
Imat_data = torch.zeros(13, n + 1, dtype=torch.float32)
for i in range(len(provdatalist)):
    Imat_data[i, :] = torch.tensor(provdatalist[i].numactive)

del A, df, poplist, provlist, t, i, first, T, provdatalist
# L=G^TG, wher L is the laplacian of the graph and G is the incidence matrix
# nodes are assigned in alphabetical order: (0) Alberta, (1) British Columbia, (2) Manitoba, (3) New Brunswick, 
# (4) Newfoundland and Labrador, (5) northwest territories, (6) nova scotia, (7) Nunavut, (8) Ontario, 
# (9) Prince Edward Island, (10) Quebec, (11) Saskatchewan, (12) Yukon
# propigation through graph u_t=-Lu\

# create fully connected Laplacian
L = np.ones([13, 13]) * -1 + np.diag(np.repeat(14, 13))

# disconnect some nodes

# AB connected with BC, YT, NWT, SK, MB, ON, QC, NS, PEI
L[0, 0] = 9
L[0, 3] = 0
L[0, 4] = 0
L[0, 7] = 0

# BC connected with YT, AB, NWT, SK, MB, ON, QC
L[1, 1] = 7
L[1, 3] = 0
L[1, 4] = 0
L[1, 6] = 0
L[1, 7] = 0
L[1, 9] = 0

# MB connected with AB, BC, NU, ON, QU, SK
L[2, 2] = 6
L[2, 3] = 0
L[2, 4] = 0
L[2, 5] = 0
L[2, 6] = 0
L[2, 9] = 0
L[2, 12] = 0

# NB connected with ON, QC, NL, NS, PEI
L[3, 3] = 5
L[3, 0] = 0
L[3, 1] = 0
L[3, 2] = 0
L[3, 5] = 0
L[3, 7] = 0
L[3, 11] = 0
L[3, 12] = 0

# NL connected with NB, NS, QC, NL
L[4, 4] = 4
L[4, 0] = 0
L[4, 1] = 0
L[4, 2] = 0
L[4, 5] = 0
L[4, 7] = 0
L[4, 9] = 0
L[4, 11] = 0
L[4, 12] = 0

# Northwest territories connected with yt, bc, ab, nu, sk
L[5, 5] = 5
L[5, 10] = 0
L[5, 9] = 0
L[5, 8] = 0
L[5, 6] = 0
L[5, 4] = 0
L[5, 3] = 0
L[5, 2] = 0

# nova scotia connected with NL, QC, NB, PEI, AB, ON
L[6, 6] = 6
L[6, 1] = 0
L[6, 2] = 0
L[6, 5] = 0
L[6, 7] = 0
L[6, 11] = 0
L[6, 12] = 0

# NU connected with NWT, MB, ON, QC
L[7, 7] = 4
L[7, 12] = 0
L[7, 11] = 0
L[7, 9] = 0
L[7, 6] = 0
L[7, 4] = 0
L[7, 3] = 0
L[7, 1] = 0
L[7, 0] = 0

# ON connected with AB, BC, MB, NB, NL, NS, NU, PEI, QC, SK
L[8, 8] = 10
L[8, 5] = 0
L[8, 12] = 0

# PEI connected with AB, ON, NS, QC, NB
L[9, 9] = 5
L[9, 1] = 0
L[9, 2] = 0
L[9, 4] = 0
L[9, 5] = 0
L[9, 7] = 0
L[9, 11] = 0
L[9, 12] = 0

# QC connected with AB, BC, MB, NB, NL, NS, NU, ON, PEI, SK
L[10, 10] = 10
L[10, 5] = 0
L[10, 12] = 0

# SK connected with AB, BC, MB, NWT, ON, QC
L[11, 11] = 6
L[11, 3] = 0
L[11, 4] = 0
L[11, 6] = 0
L[11, 7] = 0
L[11, 9] = 0
L[11, 12] = 0

# Yukon connected with BC, AB, NWT
L[12, 12] = 3
L[12, 11] = 0
L[12, 10] = 0
L[12, 9] = 0
L[12, 8] = 0
L[12, 7] = 0
L[12, 6] = 0
L[12, 4] = 0
L[12, 3] = 0
L[12, 2] = 0
L = torch.tensor(L, dtype=torch.float32)

# test L=0 for now to simplify debugging
#L = torch.zeros(13, 13, dtype=torch.float32)


#########################################################
####################3  optimize parameters  ############
#########################################################
# define function
def SIRmodel(S0, I0, Beta, L, Gamma, dt):
    Smat = torch.zeros(13, n + 1, dtype=torch.float32)
    Imat = torch.zeros(13, n + 1, dtype=torch.float32)
    t = torch.zeros(n + 1, dtype=torch.float32)
    Beta = torch.relu(Beta)
    S = S0
    I = I0
    Smat[:, 0] = 1.0 * S
    Imat[:, 0] = 1.0 * I
    t = np.array([i * dt for i in range(n + 1)])
    for j in range(1, n + 1):
        change = Beta[:, j - 1] * (S * I)
        S = S - c_S * torch.matmul(L, S) - change
        I = I - c_I * torch.matmul(L, I) + change - Gamma * I
        Smat[:, j] = S
        Imat[:, j] = I

    return Smat, Imat, t

# assume Gamma is the same for all time and all provinces
Gamma = torch.zeros(1, dtype=torch.float32) + (0.2)
c_S = torch.zeros(13, dtype=torch.float32) + 0.0015
c_I = torch.zeros(13, dtype=torch.float32) + 0.0015
Beta = 0.24 * torch.ones(13, n + 1, dtype=torch.float32)
dt = 1.0
# step size (for gradient descent)
delta = 20e-3
deltaGamma = 0.000002
deltaCI = 0.0000001
deltaCS = 0.0000001


numiter = 40000
Imat = torch.zeros(13, n + 1, dtype=torch.float32)
print(F.mse_loss(Imat, Imat_data) / F.mse_loss(Imat_data * 0, Imat_data))

for i in range(numiter):
    #  compute function and its gradient
    I0 = Imat_data[:, 0]
    S0 = 1.0 - I0

    # Make beta a parameter that we can compute gradients 
    Beta = torch.tensor(Beta, requires_grad=True)
    Gamma = torch.tensor(Gamma, requires_grad=True)
    c_I = torch.tensor(c_I, requires_grad=True)
    c_S = torch.tensor(c_S, requires_grad=True)

    # Evaluate the objective function
    Smat, Imat, t = SIRmodel(S0, I0, Beta, L, Gamma, dt)

    # calculate loss
    loss = F.mse_loss(Imat, Imat_data) / F.mse_loss(Imat_data * 0, Imat_data)

    # Compute gradients
    loss.backward()

    # if i % 40 == 0:
    #     gradLossGamma = Gamma.grad
    #     # update the parameters
    #     with torch.no_grad():
    #         Gamma -= deltaGamma * gradLossGamma
    #     if i % 100 == 1:
    #         print(i, loss.item(), torch.norm(gradLossGamma).item())
    # elif i%40 == 1:
    #     gradLossCI = c_I.grad
    #     with torch.no_grad():
    #         c_I -= deltaCI * gradLossCI
    #     if i%100 ==1:
    #         print(i,loss.item(),torch.norm(gradLossCI).item())
    #
    # elif i % 40 == 2:
    #     gradLossCS = c_S.grad
    #     with torch.no_grad():
    #         c_S -= deltaCS * gradLossCS
    #     if i % 100 == 1:
    #         print(i, loss.item(), torch.norm(gradLossCS).item())
    #
    # else:
    gradLossBeta = Beta.grad
    # update the parameters
    with torch.no_grad():
        Beta -= delta * gradLossBeta
    if i % 100 == 1:
        print(i, loss.item(), torch.norm(gradLossBeta).item())

    Beta = torch.relu(Beta)
    # Gamma = torch.relu(Gamma)

prov_to_look = 1
pp.plot(t, Imat.detach()[prov_to_look, :], t, Imat_data[prov_to_look, :])