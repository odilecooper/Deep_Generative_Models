from opcode import opname
import os
import numpy as np
import matplotlib.pyplot as plt

dataset_names = ['train', 'valid', 'test']

def generate_one_set():
    data = None
    for x  in range(5):
        for y in range(5):
            new_data = np.random.randn(400, 2) + [x*10+5, y*10+5] # 10000/25 = 400
            label = np.full((400, 1), x+y*5)
            new_data = np.concatenate([new_data, label], axis=1)
            # plt.scatter(new_data[:, 0], new_data[:, 1])
            if data is not None:
                data = np.concatenate([data, new_data])
            else:
                data = new_data
    plt.figure(figsize=(16, 12))
    plt.scatter(data[:, 0], data[:, 1], c=data[:, 2])
    plt.savefig('data/sample.png')
    # plt.show()
    return data

def generate_all():
    for set_name in dataset_names:
        data = generate_one_set()
        set_dir = os.path.join('data/'+set_name+'.txt')
        with open(set_dir, 'w') as f:
            for i in range(data.shape[0]):
                f.write(str(data[i,0])+','+str(data[i,1])+','+str(data[i,2])+'\n')

def load_data(filename, plot=False):
    data = []
    with open(filename,'r') as file:
        for line in file.readlines():
            data.append([float(i) for i in line.split(',')])
    data = np.asarray(data)
    feat = np.array(data[:, 0:2])
    label = np.array(data[:, 2])

    if plot is True:
        plt.figure(figsize=(16, 12))
        plt.scatter(data[:, 0], data[:, 1], c=data[:, 2])
        # plt.savefig('data/train.png')
        # plt.show()
        plt.close()
    # plot density
        density = np.zeros([50, 50])
        for i in range(data.shape[0]):
            xi = int(data[i, 0])
            yi = int(data[i, 1])
            density[xi, yi] += 1
    
        fig, ax = plt.subplots()
        im = ax.imshow(density)
        ax.set_xticks(np.arange(50, step=5))
        ax.set_yticks(np.arange(50, step=5), np.flip(np.arange(50, step=5), axis=0))
        ax.invert_yaxis()
        im = ax.imshow(density, cmap=plt.cm.OrRd)
        plt.colorbar(im)
        plt.savefig('data/density.png')
        plt.close()

    # return feat, label
    return data

# generate_all()
load_data('data/train.txt', plot=True)
