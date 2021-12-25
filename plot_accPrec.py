import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def load_dict_from_file(filename):
    f = open(filename,'r')
    data=f.read()
    f.close()
    return eval(data)


def plot_global_accuracy(acc):
    fig, axs = plt.subplots(2,2)
    fig.suptitle('Global accuracy (GA) for all Leads combinations')

    # matplotlib histogram
    axs[0,0].set_title('Accuracy histogram for the Chinos set')
    axs[0,0].hist(acc.T[1], color = 'blue', edgecolor = 'black', bins = 15)
    axs[0,0].set(xlabel='Accuracy', ylabel='N leads combinations')

    axs[0,1].set_title('Accuracy histogram for the Test set')
    axs[0,1].hist(acc.T[0], color = 'blue', edgecolor = 'black', bins = 15)
    axs[0,1].set(xlabel='Accuracy (%)', ylabel='N leads combinations')

    #axs[1,0].set_title('Box plots')
    axs[1,0].boxplot((acc.T[0], acc.T[1]), notch=True, sym="o", labels=["Test set", "Chinos Set"])
    axs[1,0].set(ylabel='Accuracy')

    plt.savefig('global_accuracy.ok.png')

    plt.show()


d = load_dict_from_file("dictScores.txt")
acc = np.array(list(d.values()))

prec2id = {"I":0,"II":1,"III":2,"aVR":3,"aVL":4,"aVF":5,"V1":6,"V2":7,"V3":8,"V4":9,"V5":10,"V6":11}

plot_global_accuracy(acc)

# Mejores precordiales para los chinos y test: ojo cambiar el if para cambiar la grafica
dap = {}
Lmax = []
for cp in d:
    (at, ac) = d[cp]
    if ac >= 0.895 :
        Lmax.append(cp)
    if '_' in cp:
        precs = cp.split('_')
        for p in precs:
            if p in dap:
                dap[p].append(d[cp])
            else:
                dap[p] = [d[cp]]
    elif cp in dap:
        dap[cp].append(d[cp])
    else:
        dap[cp] = [d[cp]]
        

fig = plt.figure()

bp = []
for pm in Lmax:
  bp.append(d[pm])

bacc = np.array(bp)
y_pos = range(len(Lmax))
plt.xlabel('Leads combinations')
plt.ylabel('Accuracy')
plt.title('Best Leads Combinations (Test-Set 1: Chinos)')

#CAmbiar el indice 0/1 y el nombre de la grafica
plt.plot(y_pos, bacc.T[1], color='gray', marker='o', linestyle='dashed', linewidth=1, markersize=8)
#plt.plot(y_pos, bacc.T[1], color='brown', marker='o', linestyle='dashed', linewidth=1, markersize=8, label='Chinos Test-Set')
#plt.legend()
plt.xticks(y_pos, Lmax, rotation=90, size=8)
plt.tight_layout()
plt.savefig('bestLeads4Chinos.png')
plt.show()

####################################################################

fig = plt.figure()
plt.title('Leads Accuracy for Test-Set 2 (Clinic)')
Ldata, Llabels = [], []

for p in dap:
    acc = np.array(dap[p])   
    Ldata.append(acc.T[0])
    Llabels.append(p)

plt.ylabel('Accuracy')
plt.boxplot(Ldata, notch=True, sym="o",  labels=Llabels, showmeans=True, meanline=True, showfliers=False)
plt.savefig('leadsAcc4Clinic.png')    
   
# plot_global_accuracy(acc)
plt.show()



"""
data = acc.T[1]
x, mean, std = np.linspace(0,1,100), data.mean(), data.std()
y = norm.pdf(x,mean,std)
axs[0,0].plot(x,y,color='coral')

data = acc.T[0]
x, mean, std = np.linspace(0,1,100), data.mean(), data.std()
y = norm.pdf(x,mean,std)
axs[0,0].plot(x,y)
"""


# Hide x labels and tick labels for top plots and y ticks for right plots.
#for ax in axs.flat:
#    ax.label_outer()



