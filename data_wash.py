import os
from sklearn.model_selection import KFold
path_0 = r"./citeulike_a/"
path_1 = r"./citeulike_a/"
filelist = os.listdir(path_0)
import random
import numpy as np
ROW_MUL=16981
COL_MUL=20000

ROW_USER=5551
COL_USER=16981


def dat2csv():

    for files in filelist:

        dir_path = os.path.join(path_0, files)
        # 分离文件名和文件类型
        file_name = os.path.splitext(files)[0]  # 文件名
        file_type = os.path.splitext(files)[1]  # 文件类型

        if file_type==".dat" and file_name=="mult":#词袋数据处理
            print("find it mult.dat!")

            file_test = open(dir_path, 'r')
            # 将.dat文件转为.csv文件
            new_dir = os.path.join(path_1, str(file_name) + '.csv')

            file_test2 = open(new_dir, 'w')

            for lines in file_test.readlines():
        #现在需要根据数据把他们转进一个巨大的矩阵中，矩阵的元素为0或者1
                for i in range(len(lines.split(' '))):
                    lines.split(' ')[i]=int(lines.split(' ')[i])
                str_data = ",".join(lines.split(' '))

                file_test2.write(str_data)


        if file_type == ".dat" and file_name == "users":#用户数据处理

            file_test = open(dir_path, 'r')  # 以只读方式打开源文件
            # 将.dat文件转为.csv文件
            new_dir = os.path.join(path_1, str(file_name) + '.csv')

            file_test2 = open(new_dir, 'w')

            count = 0
            for lines in file_test.readlines():
                # 现在需要根据数据把他们转进一个巨大的矩阵中，矩阵的元素为0或者1
                for i in range(len(lines.split(' '))):
                    lines.split(' ')[i] = (lines.split(' ')[i])
                str_data = ",".join(lines.split(' '))

                if (count == ROW_USER - 1):
                    file_test2.write(str_data+"\n")
                else:
                    file_test2.write(str_data)
                count=count+1


            print("proprocess_user_fin!")
            break

def csv2fin():#完成词袋数据的预处理

    for files in filelist:
        dir_path = os.path.join(path_0, files)
        # 分离文件名和文件类型
        file_name = os.path.splitext(files)[0]  # 文件名
        file_type = os.path.splitext(files)[1]  # 文件类型

        if file_type == ".csv" and file_name == "mult":

            item_info = [[0 for j in range(COL_MUL)] for i in range(ROW_MUL)]  # i:user,j:item
            matrix_temp = []

            print("find multi.csv!")
            file_test = open(dir_path, 'r')
            new_dir = os.path.join(path_1, str(file_name) + '.txt')#二次修改数据格式
            file_final = open(new_dir, 'w')

            count=0
            for lines in file_test.readlines():
                print(count)

                matrix_temp=lines.split(',') #csv信息使用逗号分隔
                num=int(matrix_temp[0])#获得词袋数目（重复的不算入在内）
                for k in range(num):

                    index,freq=matrix_temp[k+1].split('_')#下标与频数
                    item_info[int(count)][int(index)-1]=int(freq)
                    freq=int(freq)

                    if(count!=0 or (count==0 and k!=0)):#特殊行处理换行符以及tab
                        file_final.write('\n' + str(count) + "\t" + str(int(index)-1) + "\t" + str(freq))
                    else:
                        file_final.write(str(count) + "\t" + str(int(index)-1) + "\t" + str(freq))
                count=count+1

            file_final.close()
            file_test.close()


        if file_type == ".csv" and file_name == "users":

            user_ratings = [[0 for j in range(COL_USER)] for i in range(ROW_USER)]  # i:user,j:item
            matrix_temp = []

            with_info=[0 for j in range(ROW_USER)]#用于存储user感兴趣的物品总数，辅助计算recall值

            file_test = open(dir_path, 'r')
            new_dir = os.path.join(path_1, str(file_name) + '.txt')  # 二次修改数据格式
            file_final = open(new_dir, 'w')

            count = 0
            for lines in file_test.readlines():
                matrix_temp = lines.split(',')
                if (count == ROW_USER - 2):
                    print(matrix_temp)
                if(count==ROW_USER-1):
                    print(matrix_temp)

                num = int(matrix_temp[0])
                with_info[count]=num
                for k in range(num):
                    index= matrix_temp[k + 1]
                    user_ratings[int(count)][int(index)] =1

                count = count + 1

            for i in range(ROW_USER):
                for j in range(COL_USER):

                    if user_ratings[i][j]==1:
                        if(j == 0 and i==0):
                            file_final.write(str(with_info[i]) + "\t" + str(i + 1) + "\t" + str(j + 1) + "\t" + str(user_ratings[i][j]))

                        elif(j==COL_USER-1 and i==ROW_USER-1):
                            file_final.write('\n' + str(with_info[i]) + "\t" + str(i + 1) + "\t" + str(j + 1) + "\t" + str(
                                user_ratings[i][j])+'\n')
                        else:
                            file_final.write('\n' + str(with_info[i])+ "\t" +str(i+1) + "\t" + str(j+1) + "\t" + str(user_ratings[i][j]))

            file_final.close()
            file_test.close()

def create_train_test():#随机打乱抽取形成训练集与测试集(使用交叉验证）
    with open('./data/citeulike_a/mult.txt') as f:
        content = f.readlines()
    random.shuffle(content)
    new_dir = os.path.join(path_1, str('mult_shuffle') + '.txt')  # 二次修改数据格式
    file_final = open(new_dir, 'w')
    for j in range(len(content)):
        file_final.write(content[j])
    file_final.close()

    with open('./data/citeulike_a/users.txt') as f:
        content = f.readlines()
    random.shuffle(content)
    new_dir = os.path.join(path_1, str('users_shuffle') + '.txt')  # 二次修改数据格式
    file_final = open(new_dir, 'w')
    count=0
    for j in range(len(content)):

        file_final.write(content[j])
        count=count+1
    file_final.close()


def KFOLD_5():#划分为5重交叉验证集
    kf=KFold(n_splits=5,shuffle=True)
    filename=path_0+'users_shuffle.txt'
    f= open(filename, 'r')

    fold_temp=[]

    for lines in f.readlines():
        fold_temp.append(lines)

    count=0
    for train_index,test_index in kf.split(fold_temp):
        print(train_index)

        t_name="train_fold_"+str(count)
        t_dir=os.path.join(path_1, str(t_name) + '.txt')
        file_tr=open(t_dir, 'w')

        fold_temp = np.array(fold_temp)
        X_train, X_test = fold_temp[train_index], fold_temp[test_index]

        for k in range(len(X_train)):

            file_tr.write(X_train[k])

        file_tr.close()

        t_name = "test_fold_" + str(count)
        t_dir = os.path.join(path_1, str(t_name) + '.txt')
        file_te = open(t_dir, 'w')

        for k in range(len(X_test)):
            file_te.write(X_test[k])

        file_te.close()

        count=count+1

def choose_p():#用于形成dense p 数据集，筛选相关的p个词形成train，剩下的为test
    for files in filelist:
        dir_path = os.path.join(path_0, files)
        # 分离文件名和文件类型
        file_name = os.path.splitext(files)[0]  # 文件名
        file_type = os.path.splitext(files)[1]  # 文件类型

        if file_type == ".csv" and file_name == "users":

            user_ratings = [[0 for j in range(COL_USER)] for i in range(ROW_USER)]  # i:user,j:item
            matrix_temp = []

            with_info = [0 for j in range(ROW_USER)]  # 用于存储user感兴趣的物品总数，辅助计算recall值

            print("find users.csv!")
            file_test = open(dir_path, 'r')


            t_name = "train_fold_101"
            t_dir = os.path.join(path_1, str(t_name) + '.txt')
            file_tr = open(t_dir, 'w')

            t_name = "test_fold_101"
            t_dir = os.path.join(path_1, str(t_name) + '.txt')
            file_te = open(t_dir, 'w')

            train_list=[]
            test_list=[]
            count=0
            for lines in file_test.readlines():

                matrix_temp = lines.split(',')
                if (count == ROW_USER - 2):
                    print(matrix_temp)
                if (count == ROW_USER - 1):
                    print(matrix_temp)

                num = int(matrix_temp[0])
                with_info[count] = num
                if num<10:
                    for j in range(1,num+1):
                        if (j == 1 and count == 0):
                            file_tr.write(str(with_info[count]) + "\t" + str(count) + "\t" + str(int(matrix_temp[j])) + "\t" + str(1))
                        elif (j == num and count == ROW_USER - 1):
                            file_tr.write(
                                '\n' + str(with_info[count]) + "\t" + str(count) + "\t" + str(int(matrix_temp[j])) + "\t" + str(1) + '\n')
                        else:
                            file_tr.write(
                                '\n' + str(with_info[count]) + "\t" + str(count) + "\t" + str(int(matrix_temp[j])) + "\t" + str(1))



                else:
                    mid=matrix_temp[1:]
                    random.shuffle(mid)
                    for j in range(10):
                        if (j == 0 and count == 0):
                            file_tr.write(str(with_info[count]) + "\t" + str(count) + "\t" + str(int(mid[j])) + "\t" + str(1))
                        elif (j == 9 and count == ROW_USER - 1):
                            file_tr.write(
                                '\n' + str(with_info[count]) + "\t" + str(count) + "\t" + str(int(mid[j])) + "\t" + str(1) + '\n')
                        else:
                            file_tr.write(
                                '\n' + str(with_info[count]) + "\t" + str(count) + "\t" + str(int(mid[j])) + "\t" + str(1))

                    for j in range(10,num):
                        if (j == 10 and count == 0):
                            file_te.write(str(with_info[count]) + "\t" + str(count) + "\t" + str(int(mid[j])) + "\t" + str(1))
                        elif (j == num and count == ROW_USER - 1):
                            file_te.write(
                                '\n' + str(with_info[count]) + "\t" + str(count) + "\t" + str(int(mid[j])) + "\t" + str(1) + '\n')
                        else:
                            file_te.write(
                                '\n' + str(with_info[count]) + "\t" + str(count) + "\t" + str(int(mid[j])) + "\t" + str(1))



                count = count + 1

            file_te.close()
            file_tr.close()
            file_test.close()

if __name__=="__main__":
    # dat2csv()
    # csv2fin()
    # choose_p()
    with open('./data/citeulike_t/mult.txt') as f:
        content = f.readlines()
    random.shuffle(content)
    new_dir = os.path.join(path_1, str('mult_shuffle') + '.txt')  # 二次修改数据格式
    file_final = open(new_dir, 'w')
    for j in range(len(content)):
        file_final.write(content[j])
    file_final.close()
