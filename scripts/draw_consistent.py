import matplotlib.pyplot as plt
import numpy as np
import math
def consist_bar():
    name_list = []
    data = np.load('F:\datasets\IEMOCAP\prob1.npy')
    easy = []
    mid = []
    hard = []
    for i in range(len(data)):
        if data[i] == 1.0:
            easy.append(data[i])
        elif data[i] >= 0.65 and data[i] < 1.0:
            mid.append(data[i])
        else:
            hard.append(data[i])
    name_list = ['A', 'B', 'C']
    num_list = [len(easy), len(mid), len(hard)]
    x = range(len(num_list))
    plt.bar(x, num_list, color='#CD69C9',label='', tick_label=name_list)
    plt.ylabel('utterances')
    # ?\plt.xlabel('probability distribution of emotional consistency')
    # p1 = plt.bar(x, height=num_list, width=0.5, label=['A','B','C'], tick_label=name_list)
    plt.legend(loc="upper right")
    # for a, b in zip(x, num_list):
    #     plt.text(a, b + 0.05, '' , ha='center', va='bottom', fontsize=10)
    plt.savefig('C:/Users/Administrator/Desktop/ABC.pdf')
    print()
    plt.show()

def bar():
    name_list = ['Monday', 'Tuesday', 'Friday', 'Sunday']
    num_list = [1.5, 0.6, 7.8, 6]
    num_list1 = [1, 2, 3, 1]
    x = list(range(len(num_list)))
    total_width, n = 0.8, 2
    width = total_width / n

    plt.bar(x, num_list, width=width, label='boy', fc='y')
    for i in range(len(x)):
        x[i] = x[i] + width
    plt.bar(x, num_list1, width=width, label='girl', tick_label=name_list, fc='r')
    plt.legend()
    plt.show()

def example():
    name = ["nue", "ang", "hap", "sad"]
    y1 = [333, 303, 273, 288]
    y2 = [1153, 676, 1096, 598]
    y3 = [222, 124, 267, 198]

    x = np.arange(len(name))
    width = 0.25

    plt.bar(x, y1, width=width, label='A', color='#990099')
    plt.bar(x + width, y2, width=width, label='B', color='#99FFCC', tick_label=name)
    plt.bar(x + 2 * width, y3, width=width, label='C', color='#87CEFA')

    # 显示在图形上的值
    for a, b in zip(x, y1):
        plt.text(a, b + 0.1, b, ha='center', va='bottom')
    for a, b in zip(x, y2):
        plt.text(a + width, b + 0.1, b, ha='center', va='bottom')
    for a, b in zip(x, y3):
        plt.text(a + 2 * width, b + 0.1, b, ha='center', va='bottom')

    plt.xticks()
    plt.legend(loc="upper right")  # 防止label和图像重合显示不出来
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.ylabel('numbers')
    plt.xlabel('classes')
    plt.rcParams['savefig.dpi'] = 300  # 图片像素
    plt.rcParams['figure.dpi'] = 300  # 分辨率
    plt.rcParams['figure.figsize'] = (15.0, 8.0)  # 尺寸
    plt.title("merge")
    # plt.savefig('D:\\result.png')
    plt.show()

def prob():
    path='./prob.npy'
    x = np.load(path)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    ax.hist(x, bins=20,align='right',color = '#8A2BE2')
    plt.tick_params(axis='x', pad=0.1,labelsize=18)
    plt.axvline(x=0.65, ls="-", c="red")  # 添加垂直直线
    plt.text(0.58, 395, 'x=0.65')
    font2 = {'family': 'Times New Roman',
             'size': 20,
             }
    # ax.set_xlabel('probability')
    plt.xlabel('vote proportion', font2)
    plt.ylabel('utterances', font2)
    # ax.set_ylabel('utterances')
    plt.tight_layout()
    plt.savefig('D:\\prob.pdf')
    fig.show()

def example1():
    x = ["neu", "ang", "hap", "sad"]
    y1 = [333+222, 303+124, 273+267, 288+198]
    y2 = [1153+333+222, 676+303+124, 1096+273+267, 598+288+198]
    y3 = [222, 124, 267, 198]

    plt.bar(x, y2, width=0.6, label="B:0.65≤p<1", alpha=1, color='#CCFFCC',edgecolor='black',linestyle=':')
    plt.bar(x, y1, width=0.6, label="A:p=1", alpha=1,color='Silver',edgecolor='black',linestyle='--')

    plt.bar(x, y3, width=0.6, label="C:0<p<0.65",alpha=1, color='#FFCCCC',edgecolor='black')

    plt.xticks(np.arange(len(x)), x, rotation=0, fontsize=18)  # 数量多可以采用270度，数量少可以采用340度，得到更好的视图
    plt.legend(loc="upper right")  # 防止label和图像重合显示不出来
    # plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    font2 = {'family': 'Times New Roman',
             'size': 20,
             }
    plt.ylabel('utterances', font2)
    plt.xlabel('label emotions', font2)
    # plt.rcParams['savefig.dpi'] = 300  # 图片像素
    # plt.rcParams['figure.dpi'] = 300  # 分辨率
    plt.rcParams['figure.figsize'] = (20.0, 8.0)  # 尺寸
    # plt.title("title")
    plt.tight_layout()
    plt.savefig('D:\\label.pdf')
    plt.show()

def get_function():
    x = np.linspace(0,1, 100)
    y = (1 - math.pow(i, 0.7) for i in x)
    fu = [j/ 0.3 for j in y]
    y1 = (1 - math.pow(i, 0.3) for i in x)
    fu1 = [j / 0.3 for j in y1]
    plt.plot(x, fu, 'r',x,fu1,'b',linewidth=2)
    plt.legend(loc="upper left")
    plt.show()

if __name__ == '__main__':
    prob()