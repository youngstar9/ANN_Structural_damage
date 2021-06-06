#%%
import numpy as np 
import pandas as pd 
import seaborn as sns
# import matplotlib.pyplot as plt
# plt.rcParams['font.sans-serif'] = ['Times New Roman']
# plt.rcParams['axes.unicode_minus'] = False

import tensorflow as tf 
from tensorflow import keras
from sklearn.metrics import confusion_matrix
from tensorflow.keras import losses,datasets, layers, optimizers, Sequential, metrics,models
import  os
gpu = tf.config.experimental.list_physical_devices(device_type='GPU') 
tf.config.experimental.set_memory_growth(gpu[0], True)  

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
Filepath = "sldfa"

#%%
df = pd.read_excel(r'G:\Subject\小论文2\ml\test.xlsx')
cols = ['story','ld','drift','f','a']
cs = df[cols]
y = df['D']

cs_stats = cs.describe()
cs_stats = cs_stats.transpose()
# print(cs_stats.head(10))
def norm(x):
    #return x
    return (x-cs_stats['mean']) / cs_stats['std']
cs_normed = norm(cs)
model_loaded = tf.keras.models.load_model(r'G:\Subject\小论文2\ml\{}\model'.format(Filepath))
print(model_loaded.evaluate(cs_normed,y))
# y = model_loaded.predict(cs)
y_pre = pd.DataFrame(model_loaded.predict(cs_normed))
y_pre.to_csv(r'G:\Subject\小论文2\ml\{}\y_pre.csv'.format(Filepath))

#%%

# plt.scatter(y_pre,y, s=1) 
# plt.show()

# import matplotlib.pyplot as plt

# plt.rcParams['font.sans-serif'] = ['SimSun']
# plt.rcParams['axes.unicode_minus'] = False

import matplotlib.pyplot as plt
from matplotlib import rcParams

config = {
    "font.family":'serif',
    "font.size": 10.5,
    "mathtext.fontset":'stix',
    "font.serif": ['Times New Roman'],
}
rcParams.update(config)

y_pre = np.loadtxt(r'G:\Subject\小论文2\ml\{}\y_pre.csv'.format(Filepath),skiprows=1,delimiter=',',usecols=(1),unpack=True)

def Curve_Fitting(x,y,deg): 
    parameter = np.polyfit(x, y, deg)    #拟合deg次多项式 
    p = np.poly1d(parameter)             #拟合deg次多项式 
    print(p)
    aa=''                               #方程拼接  —————————————————— 
    for i in range(deg+1):  
        bb=round(parameter[i],2) 
        if bb>0: 
            if i==0: 
                bb=str(bb) 
            else: 
                bb='+'+str(bb) 
        else: 
            bb=str(bb) 
        if deg==i: 
            aa=aa+'+'+bb 
        else: 
            aa=aa+bb+'x^'+str(deg-i)    #方程拼接  —————————————————— 
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.figure()

    plt.scatter(x, y,marker='.',s=5,color='gray',label='$\mathregular{R^2}$ = 0.956')     #原始数据散点图 
    # 坐标轴的刻度设置向内(in)或向外(out)

    plt.grid(linestyle="--") 
    font_dict = {'family': 'SimSun',   # serif
        'style': 'normal',   # 'italic',
        'weight': 'normal',
        'size': 10.5,
        }
    plt.xlabel("预测值",font_dict)
    plt.ylabel("真实值",font_dict)
    yy = plt.plot(x, p(x),color='k',label='y = 0.996x - 0.006')  # 画拟合曲线 
    plt.text(-1,0,aa,fontdict={'size':'1','color':'b'}) 
    #plt.legend([aa])   #拼接好的方程和R方放到图例 
    
    
    leg = plt.legend(loc=4,markerscale=0.00001)
    leg.get_frame().set_linewidth(0.0)

    plt.xlim(0, 1.2)
    plt.ylim(0, 1.2)
    plt.show() 
    print('曲线方程为：',aa) 
    print('     r^2为：',round(np.corrcoef(y, p(x))[0,1]**2,3)) 
Curve_Fitting(y_pre,y,1)

#%%
# y = np.loadtxt(r'G:\Subject\1.1\paint\2.1\y.csv',skiprows=1,delimiter=',',usecols=(1),unpack=True)
# y = y.values.tolist()
# y_pre = y_pre.values.tolist()
y = df['D']
y = y.values.tolist()
for i in range(len(y)):
    if y[i] < 0.11:
        y[i] = 1
    elif 0.11 <= y[i] and y[i] < 0.4:
        y[i] = 2
    elif 0.4 <= y[i] and y[i] < 0.77:
        y[i] = 3
    # elif 0.4 <= y[i] and y[i] < 1:
    #     y[i] = 4
    else:
        y[i] =4
# y = y.astype(np.int8)
np.savetxt(r'G:\Subject\小论文2\ml\{}\tr_label.txt'.format(Filepath),y,fmt='%i')

y_pre = np.loadtxt(r'G:\Subject\小论文2\ml\{}\y_pre.csv'.format(Filepath),skiprows=1,delimiter=',',usecols=(1),unpack=True)

# for i in range(len(y_pre)):
#     if y_pre[i] < 0.1:
#         y_pre[i] = 1
#     elif 0.1 <= y_pre[i] and y_pre[i] < 0.25:
#         y_pre[i] = 2
#     elif 0.25 <= y_pre[i] and y_pre[i] < 0.4:
#         y_pre[i] = 3
#     elif 0.4 <= y_pre[i] and y_pre[i] < 1:
#         y_pre[i] = 4
#     else:
#         y_pre[i] =5

for i in range(len(y_pre)):
    if y_pre[i] < 0.11:
        y_pre[i] = 1
    elif 0.11 <= y_pre[i] and y_pre[i] < 0.4:
        y_pre[i] = 2
    elif 0.4 <= y_pre[i] and y_pre[i] < 0.77:
        y_pre[i] = 3
    else:
        y_pre[i] = 4
# y_pre = y_pre.astype(np.int8)
np.savetxt(r'G:\Subject\小论文2\ml\{}\pr_label.txt'.format(Filepath),y_pre,fmt='%i')
#%%
#绘图


labels = ['1','2','3','4']
y_true = np.loadtxt(r'G:\Subject\小论文2\ml\{}\tr_label.txt'.format(Filepath))
y_pred = np.loadtxt(r'G:\Subject\小论文2\ml\{}\pr_label.txt'.format(Filepath))


tick_marks = np.array(range(len(labels))) + 0.5
# print(tick_marks)
def plot_confusion_matrix(tp,cm,title='Confusion Matrix',cmap=plt.cm.Blues):
    sns.set(style="darkgrid")
    plt.imshow(cm,interpolation='nearest',cmap=cmap)    #在特定窗口上显示图像
    #plt.title(title)  
    #plt.colorbar()
    
    
    xlocations = np.array(range(len(labels)))
    plt.xticks(xlocations,labels,rotation=0)
    plt.yticks(xlocations,labels)
    font_dict = {'family': 'SimSun',   # serif
    'style': 'normal',   # 'italic',
    'weight': 'normal',
    'size': 10.5,
    }
    plt.ylabel('真实分类',font_dict)
    plt.xlabel('预测分类',font_dict)

cm = confusion_matrix(y_true,y_pred)
totalt = sum(cm[i][i] for i in range(len(labels)))
totalp = totalt / cm.sum()
np.set_printoptions(precision=3) #输出精度
cm_normalized = cm.astype('float')/cm.sum(axis=1)[:,np.newaxis] #归一化
print(cm_normalized)
plt.figure(figsize=(12,8),dpi=200)

ind_array = np.arange(len(labels))
x, y = np.meshgrid(ind_array, ind_array)

for x_val, y_val in zip(x.flatten(), y.flatten()):
    c = cm_normalized[y_val][x_val]
    if c > 0.01:
        plt.text(x_val, y_val, "%0.3f" % (c,), color='black', fontsize=10, va='center', ha='center')
# offset the tick
plt.gca().set_xticks(tick_marks, minor=True)
plt.gca().set_yticks(tick_marks, minor=True)
plt.gca().xaxis.set_ticks_position('none')
plt.gca().yaxis.set_ticks_position('none')
plt.grid(True, which='minor', linestyle='-')
plt.gcf().subplots_adjust(bottom=0.15)

#plot_confusion_matrix(totalp,cm_normalized, title='Normalized confusion matrix')
plot_confusion_matrix(totalp,cm_normalized)
print(totalp)

# show confusion matrix
#plt.savefig(r'G:\Subject\1.1\{}\confusion_matrix.eps'.format(Filepath), format='eps')
plt.show()




