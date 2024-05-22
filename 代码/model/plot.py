import matplotlib.pyplot as plt

def plot_line_chart(data, x_label, y_label, title):
    x = [item[0] for item in data]
    y = [item[1] for item in data]

    plt.plot(x, y, marker='o', linestyle='-')

    plt.xlabel(x_label, fontproperties='SimHei')
    plt.ylabel(y_label, fontproperties='SimHei')
    plt.title(title, fontproperties='SimHei')
    plt.grid(True)
    plt.show()

data = [(1, 0.5), (2, 0.5), (3, 0.333), (4, 0.667), (5, 0.5), (6, 1), (7, 0.667), (8, 1), (9, 0.5), (10, 1)]  # 数据格式为 [(x1, y1), (x2, y2), ...]
x_label = '次数'
y_label = '准确率'
title = '模型预测准确率'

plot_line_chart(data, x_label, y_label, title)