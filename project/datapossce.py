import wfdb
import pywt
import numpy as np
RATIO = 0.2


# 小波去噪预处理
def denoise(data):
    # 小波变换
    coeffs = pywt.wavedec(data=data, wavelet='db5', level=9)
    cA9, cD9, cD8, cD7, cD6, cD5, cD4, cD3, cD2, cD1 = coeffs

    # 阈值去噪
    threshold = (np.median(np.abs(cD1)) / 0.6745) * (np.sqrt(2 * np.log(len(cD1))))
    cD1.fill(0)
    cD2.fill(0)
    for i in range(1, len(coeffs) - 2):
        coeffs[i] = pywt.threshold(coeffs[i], threshold)

    # 小波反变换,获取去噪后的信号
    rdata = pywt.waverec(coeffs=coeffs, wavelet='db5')
    return rdata


# 读取心电数据和对应标签,并对数据进行小波去噪
def getDataSet(number, X_data, Y_data):
    ecgClassSet = ['N', '/', 'L','R','j','e','A','a','J','E','S','F','V','f','Q']
    # mapping = {'F': 5, 'V': 5, 'f': 5,'Q':5}

    # 读取心电数据记录
    print("正在读取 " + number + " 号心电数据...")
    record = wfdb.rdrecord('D:\\snn\\ECGPossce\\mit-bih-arrhythmia-database-1.0.0\\' + number, channel_names=['MLII'])
    data = record.p_signal.flatten()
    rdata = denoise(data=data)

    # 获取心电数据记录中R波的位置和对应的标签
    annotation = wfdb.rdann('D:\\snn\\ECGPossce\\mit-bih-arrhythmia-database-1.0.0\\' + number, 'atr')
    Rlocation = annotation.sample
    Rclass = annotation.symbol

    # 去掉前后的不稳定数据
    start = 10
    end = 5
    i = start
    j = len(annotation.symbol) - end

    while i < j:
        try:
            # Rclass[i] 是标签
            lable = ecgClassSet.index(Rclass[i])
            # if Rclass[i] in mapping:
            #     lable = mapping[Rclass[i]]
            x_train = rdata[Rlocation[i] - 99:Rlocation[i] + 201]
            # 计算 R-R 间隔，并转换为 10 位二进制
            prev_diff = Rlocation[i] - Rlocation[i - 1]
            next_diff = Rlocation[i + 1] - Rlocation[i]

            prev_bin = list(map(int, format(prev_diff, '010b')))  # 转换成 10 位二进制
            next_bin = list(map(int, format(next_diff, '010b')))

            prev_bin = list(map(float, prev_bin))  # 每一位转换成 0.0 或 1.0
            next_bin = list(map(float, next_bin))  # 每一位转换成 0.0 或 1.0

            # 扩展 x_train
            x_train = list(x_train) + prev_bin + next_bin
            if len(x_train) != 320:
                print(f"警告: 样本 {i} 长度错误: {len(x_train)}，跳过该样本")
                print(x_train)
                print(prev_diff)
                print(prev_bin)
                print(next_diff)
                print(next_bin)
                i += 1
                continue
            X_data.append(x_train)
            Y_data.append(lable)
            i += 1
        except ValueError:
            i += 1
    return


# 加载数据集并进行预处理
def loadData():
    numberSet = ['100', '101', '103', '105', '106', '107', '108', '109', '111', '112', '113', '114', '115',
                 '116', '117', '119', '121', '122', '123', '124', '200', '201', '202', '203', '205', '208',
                 '210', '212', '213', '214', '215', '217', '219', '220', '221', '222', '223', '228', '230',
                 '231', '232', '233', '234']
    dataSet = []
    lableSet = []
    for n in numberSet:
        getDataSet(n, dataSet, lableSet)

    # 转numpy数组,打乱顺序
    dataSet = np.array(dataSet).reshape(-1, 320)
    lableSet = np.array(lableSet).reshape(-1, 1)
    train_ds = np.hstack((dataSet, lableSet))
    np.random.shuffle(train_ds)

    # 数据集及其标签集
    X = train_ds[:, :320].reshape(-1, 320, 1)
    Y = train_ds[:, 320]

    # 测试集及其标签集
    shuffle_index = np.random.permutation(len(X))
    # 设定测试集的大小 RATIO是测试集在数据集中所占的比例
    test_length = int(RATIO * len(shuffle_index))
    # 测试集的长度
    test_index = shuffle_index[:test_length]
    # 训练集的长度
    train_index = shuffle_index[test_length:]
    X_test, Y_test = X[test_index], Y[test_index]
    X_train, Y_train = X[train_index], Y[train_index]
    return X_train, Y_train, X_test, Y_test


def main():
    # X_train,Y_train为所有的数据集和标签集
    # X_test,Y_test为拆分的测试集和标签集
    X_train, Y_train, X_test, Y_test = loadData()
    print(X_train.shape)
    # 输出类别统计信息
    print("\n训练集类别统计:")
    unique_train, counts_train = np.unique(Y_train, return_counts=True)
    for cls, cnt in zip(unique_train, counts_train):
        print(f"类别 {cls}: {cnt} 个样本")

    print("\n测试集类别统计:")
    unique_test, counts_test = np.unique(Y_test, return_counts=True)
    for cls, cnt in zip(unique_test, counts_test):
        print(f"类别 {cls}: {cnt} 个样本")
    train_data = np.hstack((X_train.reshape(X_train.shape[0], -1), Y_train.reshape(-1, 1)))
    # 合并X_test和Y_test
    test_data = np.hstack((X_test.reshape(X_test.shape[0], -1), Y_test.reshape(-1, 1)))

    # 保存为CSV文件
    np.savetxt('train_data15(320).csv', train_data, delimiter=',')
    np.savetxt('test_data15(320).csv', test_data, delimiter=',')
    print("训练集和测试集已分别保存为 train_data.csv 和 test_data.csv")


if __name__ == '__main__':
    main()
