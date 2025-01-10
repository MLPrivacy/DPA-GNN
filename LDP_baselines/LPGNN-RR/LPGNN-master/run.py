import subprocess


dataset = ['cora']            #'cora','citeseer','lastfm','facebook', 'amazon', 'reddit'


for ds in dataset:
    if ds == 'cora':
        kx = 2
        ky = 0
    # if ds == 'citeseer':
    #     kx = 2
    #     ky = 0
    # if ds == 'lastfm':
    #     kx = 2
    #     ky = 0
    # if ds == 'facebook':
    #     kx = 0
    #     ky = 0
    # if ds == 'amazon':
    #     kx = 2
    #     ky = 0
    # if ds == 'reddit':
    #     kx = 2
    #     ky = 0

    all_eps = [10]   #2, 4, 6, 8, 10
    ey_values = [1]    #1,1,5,2
    ex_rate = [0.05]

    for eps in all_eps:
        for ey in ey_values:
            for rate in ex_rate:
                ex = eps * rate
                ee = eps - ex - ey


                cmd = f'Python ./main.py -d {ds} -ex {ex} -ey {ey} --e-eps {ee} -kx {kx} -ky {ky} -r 10 --model gcn'
                subprocess.run(cmd, shell=True)

