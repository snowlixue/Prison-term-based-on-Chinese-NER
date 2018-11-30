import json
import pandas
import numpy as np

#进行分块处理
def word_class():
    f = open("final_all_data/exercise_contest/data_valid.json", "r", encoding="utf-8")
    list = []
    texts = f.readlines()
    for text in texts:
        line = json.loads(text)
        if len(line['meta']['accusation']) <= 1:
                print(line['meta']['accusation'][0])
                if line['meta']['accusation'][0] not in list:
                    list.append(line['meta']['accusation'][0])
                filename = 'data/cnews/cnew.valid.txt'
                w = open(filename, "a", encoding="UTF-8")
                w.write(line['meta']['accusation'][0])
                w.write("     ")
                w.write(line["fact"].replace("\n","").strip().replace("\r",""))
                print(line["fact"].replace("\n","").strip())
                print(len(list))
                print(list)
                w.write("\n")
                w.close()


#提取故意伤害罪
def get_hurt():
    f = open("final_all_data/exercise_contest/data_test.json", "r", encoding="utf-8")
    list = []
    texts = f.readlines()
    for text in texts:
        line = json.loads(text)
        for i in range(0, len(line['meta']['accusation'])):
            if str(line['meta']['accusation'][i]) == '故意伤害':
                print(line['meta']['accusation'])
                filename = 'data/hurt/hurttrain.txt'
                w = open(filename, "a", encoding="UTF-8")
                w.write(line["fact"].replace("\n","").strip().replace("\r",""))
                w.write("     ")
                w.write(str(line['meta']['relevant_articles']))
                w.write("     ")
                w.write(str(line['meta']['punish_of_money']))
                w.write("     ")
                w.write(str(line['meta']['term_of_imprisonment']['imprisonment']))
                w.write("\n")
                w.close()

#提取回归信息
def get_regress():
    f1 = 'data/hurt/finall.txt'
    f2 = 'data/hurt/regress.csv'
    f = open(f1, 'r', encoding='utf-8')
    texts = f.readlines()
    w = open(f2, 'a', encoding='utf-8')
    for text in texts:
        list = text.split()
        for listt in list:
            if '/hurt' in listt:
                w.write(listt.replace('/hurt', '')+' ')
        w.write(list[-3]+' ')
        w.write(list[-2]+' ')
        w.write(list[-1]+' ')
        w.write('\n')
    w.close()
    f.close()

#统计出伤害数量
def get_hurt_numble():
    f = open('data/hurt/regress.csv', 'r', encoding='utf-8')
    texts = f.readlines()
    hurt = []
    for text in texts:
        list = text.split()
        for listt in list:
            if listt not in hurt:
                if listt != '[234]':
                    if listt[0] not in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
                        hurt.append(listt)
    print(hurt)

#获取回归csv文件
def get_line_data():
    hurt = ['轻伤二级', '轻微伤', '轻伤', '重伤', '伤残六级', '轻伤一级', '二级', '重伤二级', '轻微伤偏重', '十级伤残', '一级', '九级伤残', '轻伤（偏重）', '轻伤达九级伤残', '死亡', '七级伤残', '四级伤残']
    w = open('data/hurt/line.csv', 'a', encoding='utf-8')
    w.write('num'+' ')
    for hurtt in hurt:
        w.write(hurtt+' ')
    w.write('result')
    w.write('\n')
    f = open('data/hurt/regress.csv', 'r', encoding='utf-8')
    texts = f.readlines()
    print(texts)
    temp = 0
    for text in texts:
        #print(text)
        nums = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        lines = text.replace('\n', '').split()
        print(lines)
        for line in lines:
            for i in range(0, len(hurt)):
                if line == hurt[i]:
                    nums[i] += 1
        w.write(str(temp)+' ')
        temp += 1
        for num in nums:
            w.write(str(num)+' ')
        w.write(text[-4]+text[-3])
        w.write('\n')

#线性回归
#函数说明:加载数据
#xArr - x数据集
#yArr - y数据集
def loadDataSet(fileName):
    numFeat = len(open(fileName, encoding='utf-8').readline().split('\t')) - 2
    xArr = []; yArr = []
    fr = open(fileName, encoding='utf-8')
    next(fr)
    for line in fr.readlines():
        lineArr =[]
        curLine = line.strip()
        for i in range(1, numFeat):
            lineArr.append(float(curLine[i]))
        xArr.append(lineArr)
        yArr.append(float(curLine[-1]))
    return xArr, yArr

"""
函数说明:使用局部加权线性回归计算回归系数w
    k - 高斯核的k,自定义参数
    ws - 回归系数
"""
def lwlr(testPoint, xArr, yArr, k = 1.0):
    xMat = np.mat(xArr); yMat = np.mat(yArr).T
    m = np.shape(xMat)[0]
    weights = np.mat(np.eye((m)))                                        #创建权重对角矩阵
    for j in range(m):                                                  #遍历数据集计算每个样本的权重
        diffMat = testPoint - xMat[j, :]
        weights[j, j] = np.exp(diffMat * diffMat.T/(-2.0 * k**2))
    xTx = xMat.T * (weights * xMat)
    if np.linalg.det(xTx) == 0.0:
        print("矩阵为奇异矩阵,不能求逆")
        return
    ws = xTx.I * (xMat.T * (weights * yMat))                            #计算回归系数
    return testPoint * ws

    """
    函数说明:局部加权线性回归测试
        testArr - 测试数据集,测试集
        k - 高斯核的k,自定义参数
    Returns:
        ws - 回归系数
    """
def lwlrTest(testArr, xArr, yArr, k=1.0):
    m = np.shape(testArr)[0]                                            #计算测试数据集大小
    yHat = np.zeros(m)
    for i in range(m):                                                    #对每个样本点进行预测
        yHat[i] = lwlr(testArr[i],xArr,yArr,k)
    return yHat

    """
    函数说明:计算回归系数w
    Parameters:
        xArr - x数据集
        yArr - y数据集
    Returns:
        ws - 回归系数
    """
def standRegres(xArr,yArr):
    xMat = np.mat(xArr); yMat = np.mat(yArr).T
    xTx = xMat.T * xMat                            #根据文中推导的公示计算回归系数
    if np.linalg.det(xTx) == 0.0:
        print("矩阵为奇异矩阵,不能求逆")
        return
    ws = xTx.I * (xMat.T*yMat)
    return ws

    """
    误差大小评价函数
    Parameters:
        yArr - 真实数据
        yHatArr - 预测数据
    Returns:
        误差大小
    """
def rssError(yArr, yHatArr):
    return ((yArr - yHatArr) **2).sum()

if __name__=='__main__':
    abX, abY = loadDataSet('data/hurt/line.csv')
    print('训练集与测试集相同:局部加权线性回归,核k的大小对预测的影响:')
    yHat01 = lwlrTest(abX[0:99], abX[0:99], abY[0:99], 0.1)
    yHat1 = lwlrTest(abX[0:99], abX[0:99], abY[0:99], 1)
    yHat10 = lwlrTest(abX[0:99], abX[0:99], abY[0:99], 10)
    print('k=0.1时,误差大小为:', rssError(abY[0:99], yHat01.T))
    print('k=1  时,误差大小为:', rssError(abY[0:99], yHat1.T))
    print('k=10 时,误差大小为:', rssError(abY[0:99], yHat10.T))
    print('')
    print('训练集与测试集不同:局部加权线性回归,核k的大小是越小越好吗？更换数据集,测试结果如下:')
    yHat01 = lwlrTest(abX[100:199], abX[0:99], abY[0:99], 0.1)
    yHat1 = lwlrTest(abX[100:199], abX[0:99], abY[0:99], 1)
    yHat10 = lwlrTest(abX[100:199], abX[0:99], abY[0:99], 10)
    print('k=0.1时,误差大小为:', rssError(abY[100:199], yHat01.T))
    print('k=1  时,误差大小为:', rssError(abY[100:199], yHat1.T))
    print('k=10 时,误差大小为:', rssError(abY[100:199], yHat10.T))
    print('')
    print('训练集与测试集不同:简单的线性归回与k=1时的局部加权线性回归对比:')
    print('k=1时,误差大小为:', rssError(abY[100:199], yHat1.T))
    ws = standRegres(abX[0:99], abY[0:99])
    yHat = np.mat(abX[100:199]) * ws
    print('简单的线性回归误差大小:', rssError(abY[100:199], yHat.T.A))
