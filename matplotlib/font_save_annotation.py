import os
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.font_manager
from scipy.interpolate import make_interp_spline
# import seaborn as sns

def calc_ratio(res_file):   
    ious = np.loadtxt(res_file,delimiter=' ').reshape(-1,4)
    bpp = ious[:,0].mean()
    bpn = ious[:,1].mean()
    anp = ious[:,2].mean()
    app = ious[:,3].mean()
    return bpp, bpn, anp, app

def draw_trans_iter(root_dir):
    epochs = [x for x in range(101)]
    bpps, bpns, anps, apps = [],[],[],[]
    for epoch in epochs:
        filename = 'epoch_' + str(epoch) + '.txt'
        ious = np.loadtxt(os.path.join(root_dir, filename),delimiter=' ').reshape(-1,4)
        # import ipdb; ipdb.set_trace()
        bpp = ious[:,0].tolist()
        bpn = ious[:,1].tolist()
        anp = ious[:,2].tolist()
        app = ious[:,3].tolist()
        bpps += bpp
        bpns += bpn
        anps += anp
        apps += app
    inters = [x for x in range(len(bpps))]
    plt.figure(figsize=(19.20,10.80))
    plt.figure(1)
    ax1 = plt.subplot(221)
    plt.plot(inters,bpps)
    ax2 = plt.subplot(222)
    plt.plot(inters,bpns)
    ax3 = plt.subplot(223)
    plt.plot(inters,anps)
    ax4 = plt.subplot(224)
    plt.plot(inters,apps)
    plt.show()
    

def draw_trans_epoch(root_dir):
    files = os.listdir(root_dir)
    # import ipdb; ipdb.set_trace()
    epochs = [x for x in range(101)]
    bpps, bpns, anps, apps = [0]*101, [0]*101, [0]*101, [0]*101
    for file in files:
        epoch = eval(file.strip('epoch_').strip('.txt'))
        bpp, bpn, anp, app = calc_ratio(os.path.join(root_dir, file))
        bpps[epoch] = bpp
        bpns[epoch] = bpn
        anps[epoch] = anp
        apps[epoch] = app
    
    plt.figure(figsize=(19.20,10.80))
    plt.figure(1)
    ax1 = plt.subplot(221)
    plt.plot(epochs,bpps)
    ax2 = plt.subplot(222)
    plt.plot(epochs,bpns)
    ax3 = plt.subplot(223)
    plt.plot(epochs,anps)
    ax4 = plt.subplot(224)
    plt.plot(epochs,apps)
    plt.show()

    
    # font = {
    #     'family' : 'Times New Roman',
    # # 'weight' : 'normal',
    # 'weight' : 'light',
    # 'size'   : 25,
    # }

    
    # plt.figure(figsize=(12,9))
    # y = [0.74 for x in epochs]
    # plt.plot(epochs, y,c='r',linewidth=2, linestyle= '-.') 
    # plt.plot(epochs,bpps, c='C0')
    # plt.xlabel('Input IoU with ground-truth box',font)
    # plt.ylabel('Classification Confidence',font)
    # plt.xlim(0., 100)
    # plt.ylim(0., 1.0)
    # plt.xticks(np.arange(0., 100, 10),fontsize= 15)
    # plt.yticks(np.arange(0., 1.0, 0.1),fontsize= 15)
    # plt.annotate(r'$prob=0.74$', xy=(10, 0.74), xycoords='data', xytext=(+30, +30),
    #          textcoords='offset points', fontsize=16,
    #          arrowprops=dict(arrowstyle='->', connectionstyle="arc3,rad=.2"))
    # # plt.grid(True)
    # plt.savefig(r'C:\Users\xiaoming\Desktop\bf.png', dpi = 300)
    # plt.clf()

    # y = [0.26 for x in epochs]
    # plt.plot(epochs, y,c='r',linewidth=2, linestyle= '-.') 
    # plt.plot(epochs,anps, c='C2')
    # plt.xlabel('Input IoU with ground-truth box',font)
    # plt.ylabel('Classification Confidence',font)
    # plt.xlim(0., 100)
    # plt.ylim(0., 1.0)
    # plt.xticks(np.arange(0., 100, 10),fontsize= 15)
    # plt.yticks(np.arange(0., 1.0, 0.1),fontsize= 15)
    # plt.annotate(r'$prob=0.26$', xy=(10, 0.26), xycoords='data', xytext=(+30, +30),
    #          textcoords='offset points', fontsize=16,
    #          arrowprops=dict(arrowstyle='->', connectionstyle="arc3,rad=.2"))
    # # plt.grid(True)
    # plt.savefig(r'C:\Users\xiaoming\Desktop\af.png', dpi = 300)



if __name__ == "__main__":
    root_dir = r'C:\Users\xiaoming\Desktop\ious\neg1'
    draw_trans_epoch(root_dir)
  