import os

for i in range(70):
    os.system("python train_ASTGCN_r.py --config configurations/PEMS04_astgcn.conf")
    if i == 15:
        break