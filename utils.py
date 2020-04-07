import numpy as np
import os
from numpy import inf
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorflow.contrib.layers import batch_norm
import functools

def evaluation(test_R,test_mask_R,Estimated_R,num_test_ratings,num_true):#recall计算

    pre_numeartor1 = np.sign(Estimated_R - 0.5)
    tmp_test_R = np.sign(test_R - 0.5)

    pre_numerator2 = np.multiply(np.multiply((np.array(pre_numeartor1) == np.array(tmp_test_R)),
                                             (np.array(pre_numeartor1) == np.array(test_mask_R))),
                                 test_mask_R)  # recall需要预测为正的样本值
    numerator = np.sum(pre_numerator2)
    denominator = np.sum(np.array(tmp_test_R)==np.array(test_mask_R))
    RECALL = numerator / float(denominator)

    return RECALL#PRECISION,ACC
 #RMSE,MAE,ACC,AVG_loglikelihood,

def recall_M(num_item,num_user,m,R,Estimated_R,num_ratings):#为评分矩阵,num_true为用户所有喜欢的物品的计数值，recall m计算，top m

    num_true = [0 for i in range(num_user)]

    recall_all = 0

    for i in range(num_user):

        temp_minus = Estimated_R[i]

        before_sort_temp = []

        for o in range(num_item):
            num_true[i] = num_true[i] + R[i][o]
            if temp_minus[o] >= 0:
                before_sort_temp.append([o, temp_minus[o]])

        sort = sorted(before_sort_temp, key=lambda cus: cus[1], reverse=True)
        temp = []
        count = 0
        if len(sort) < m:
            for k in range(len(sort)):
                temp.append(sort[k][0])
        else:
            for k in range(m):
                temp.append(sort[k][0])

        if len(temp) < m:
            for d in range(len(temp)):
                if (R[i][temp[d]] == 1):
                    count = count + 1
        else:
            for d in range(m):
                if (R[i][temp[d]] == 1):
                    count = count + 1

        if num_true[i] != 0:
            recall_all = recall_all + (float(count) / num_true[i])

    recall_m = recall_all / float(num_user)

    return recall_m

def map(num_item,num_user,R,Estimated_R):#map计算


    ap_sum=0

    for i in range(num_user):

        hits = 0
        sum_precs = 0

        temp_minus = Estimated_R[i] - 0.5

        before_sort_temp = []

        for o in range(num_item):
            if temp_minus[o] >= 0:
                before_sort_temp.append([o, temp_minus[o]])
            # print(len(before_sort_temp))
        sort = sorted(before_sort_temp, key=lambda cus: cus[1], reverse=True)
        temp=[]
        for k in range(len(sort)):
            temp.append(sort[k][0])


        for t in range(len(temp)):
            if (R[i][temp[t]] == 1):
                hits += 1
                sum_precs += hits / (t + 1.0)
        if hits > 0:
            ap_sum=ap_sum+sum_precs / (np.sum(R[i]))

    return float(ap_sum)/num_user

def make_records(result_path,test_acc_list,test_precision_list,test_recall_list,test_recallAt_list,test_Map_list,map_final_list,current_time,
                 args,model_name,data_name,train_ratio,hidden_neuron,random_seed,optimizer_method,lr):#文件记录
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    overview = './results/' + 'overview.txt'
    basic_info = result_path + "basic_info.txt"
    test_record = result_path + "test_record.txt"

    with open(test_record, 'w') as g:

        g.write(str("ACC:"))
        g.write('\t')
        for itr in range(len(test_acc_list)):
            g.write(str(test_acc_list[itr]))
            g.write('\t')
        g.write('\n')

        g.write(str("RECALL:"))
        g.write('\t')
        for itr in range(len(test_recall_list)):
            g.write(str(test_recall_list[itr]))
            g.write('\t')
        g.write('\n')

        g.write(str("PRECISION:"))
        g.write('\t')
        for itr in range(len(test_precision_list)):
            g.write(str(test_precision_list[itr]))
            g.write('\t')
        g.write('\n')

        g.write(str("RECALL@M:"))
        g.write('\t')
        for itr in range(len(test_recallAt_list)):
            g.write(str(test_recallAt_list[itr]))
            g.write('\t')
        g.write('\n')

        g.write(str("Map:"))
        g.write('\t')
        for itr in range(len(test_Map_list)):
            g.write(str(test_Map_list[itr]))
            g.write('\t')
        g.write('\n')

        g.write(str("map_final_list:"))
        g.write('\t')
        for itr in range(len(map_final_list)):
            g.write(str(map_final_list[itr]))
            g.write('\t')
        g.write('\n')

    with open(basic_info, 'w') as h:
        h.write(str(args))

    with open(overview, 'a') as f:
        f.write(str(data_name))
        f.write('\t')
        f.write(str(model_name))
        f.write('\t')
        f.write(str(train_ratio))
        f.write('\t')
        f.write(str(current_time))
        f.write('\t')
        f.write(str(test_recall_list[-1]))
        f.write('\t')
        f.write(str(test_recallAt_list[-1]))
        f.write('\t')
        f.write(str(test_Map_list[-1]))
        f.write('\t')

        f.write(str(hidden_neuron))
        f.write('\t')
        f.write(str(args.corruption_level))
        f.write('\t')

        f.write(str(args.lambda_u))
        f.write('\t')
        f.write(str(args.lambda_w))
        f.write('\t')
        f.write(str(args.lambda_n))
        f.write('\t')
        f.write(str(args.lambda_v))
        f.write('\t')
        f.write(str(args.f_act))
        f.write('\t')
        f.write(str(args.g_act))
        f.write('\n')
    fig = plt.figure(figsize=(10, 6))
    Test = plt.plot(test_acc_list, '-o', markersize=1,  # 点的大小
                    markeredgecolor='black',  # 点的边框色
                    markerfacecolor='steelblue', label='Test')

    xlocator = matplotlib.ticker.MultipleLocator(20)
    ax = plt.gca()
    ax.xaxis.set_major_locator(xlocator)
    fig.autofmt_xdate(rotation=45)
    plt.xlabel('Epochs')
    plt.ylabel('ACC')
    plt.legend()
    plt.savefig(result_path + "ACC.png")
    plt.clf()

    fig = plt.figure(figsize=(10, 6))
    Test = plt.plot(test_recall_list, '-o', markersize=1,  # 点的大小
                    markeredgecolor='black',  # 点的边框色
                    markerfacecolor='steelblue', label='Test')

    xlocator = matplotlib.ticker.MultipleLocator(20)
    ax = plt.gca()
    ax.xaxis.set_major_locator(xlocator)
    fig.autofmt_xdate(rotation=45)
    plt.xlabel('Epochs')
    plt.ylabel('RECALL')
    plt.legend()
    plt.savefig(result_path + "RECALL.png")
    plt.clf()

    fig = plt.figure(figsize=(10, 6))

    Test = plt.plot(test_recallAt_list, '-o', markersize=1,  # 点的大小
                    markeredgecolor='black',  # 点的边框色
                    markerfacecolor='steelblue', label='Test')
    xlocator = matplotlib.ticker.MultipleLocator(20)
    ax = plt.gca()
    ax.xaxis.set_major_locator(xlocator)
    fig.autofmt_xdate(rotation=45)

    plt.xlabel('Epochs')
    plt.ylabel('RECALL@300')
    plt.legend()
    plt.savefig(result_path + "RECALL@300.png")
    plt.clf()

    fig = plt.figure(figsize=(10, 6))
    Test = plt.plot(test_precision_list, '-o', markersize=1,  # 点的大小
                    markeredgecolor='black',  # 点的边框色
                    markerfacecolor='steelblue', label='Test')

    xlocator = matplotlib.ticker.MultipleLocator(20)
    ax = plt.gca()
    ax.xaxis.set_major_locator(xlocator)
    fig.autofmt_xdate(rotation=45)
    plt.xlabel('Epochs')
    plt.ylabel('PRECISION')
    plt.legend()
    plt.savefig(result_path + "PRECISION.png")
    plt.clf()

    fig = plt.figure(figsize=(10, 6))
    Test = plt.plot(test_Map_list, '-o', markersize=1,  # 点的大小
                    markeredgecolor='black',  # 点的边框色
                    markerfacecolor='steelblue', label='Test')

    xlocator = matplotlib.ticker.MultipleLocator(20)
    ax = plt.gca()
    ax.xaxis.set_major_locator(xlocator)
    fig.autofmt_xdate(rotation=45)
    plt.xlabel('Epochs')
    plt.ylabel('Map')
    plt.legend()
    plt.savefig(result_path + "Map.png")
    plt.clf()

def variable_save(result_path,model_name,train_var_list1,train_var_list2,Estimated_R,test_v_ud,mask_test_v_ud):
    for var in train_var_list1:
        var_value = var.eval()
        var_name = ((var.name).split('/'))[1]
        var_name = (var_name.split(':'))[0]
        np.savetxt(result_path + var_name , var_value)

    for var in train_var_list2:
        if model_name == "DIPEN_with_VAE":
            var_value = var.eval()
            var_name = (var.name.split(':'))[0]
            print (var_name)
            var_name = var_name.replace("/","_")

            print (var.name)
            print (var_name)
            print ("================================")
            np.savetxt(result_path + var_name, var_value)
        else:
            var_value = var.eval()
            var_name = ((var.name).split('/'))[1]
            var_name = (var_name.split(':'))[0]
            np.savetxt(result_path + var_name , var_value)

    Estimated_R = np.where(Estimated_R<0.5,0,1)
    Error_list = np.nonzero( (Estimated_R - test_v_ud) * mask_test_v_ud )
    user_error_list = Error_list[0]
    item_error_list = Error_list[1]
    np.savetxt(result_path+"Estimated_R",Estimated_R)
    np.savetxt(result_path+"test_v_ud",test_v_ud)
    np.savetxt(result_path+"mask_test_v_ud",mask_test_v_ud)
    np.savetxt(result_path + "user_error_list", user_error_list)
    np.savetxt(result_path + "item_error_list", item_error_list)

def SDAE_calculate(model_name,X_c, layer_structure, W, b, batch_normalization, f_act,g_act, model_keep_prob,V_u=None):
    hidden_value = X_c
    for itr1 in range(len(layer_structure) - 1):
        ''' Encoder '''
        if itr1 <= int(len(layer_structure) / 2) - 1:
            if (itr1 == 0) and (model_name == "CDAE"):
                ''' V_u '''
                before_activation = tf.add(tf.add(tf.matmul(hidden_value, W[itr1]),V_u), b[itr1])
            else:
                before_activation = tf.add(tf.matmul(hidden_value, W[itr1]), b[itr1])
            if batch_normalization == "True":
                before_activation = batch_norm(before_activation)
            hidden_value = f_act(before_activation)
            ''' Decoder '''
        elif itr1 > int(len(layer_structure) / 2) - 1:
            before_activation = tf.add(tf.matmul(hidden_value, W[itr1]), b[itr1])
            if batch_normalization == "True":
                before_activation = batch_norm(before_activation)
            hidden_value = g_act(before_activation)
        if itr1 < len(layer_structure) - 2: # add dropout except final layer
            hidden_value = tf.nn.dropout(hidden_value, model_keep_prob)
        if itr1 == int(len(layer_structure) / 2) - 1:
            Encoded_X = hidden_value

    sdae_output = hidden_value

    return Encoded_X, sdae_output

def l2_norm(tensor):
    return tf.sqrt(tf.reduce_sum(tf.square(tensor)))

def softmax(w, t = 1.0):
    npa = np.array
    e = np.exp(npa(w) / t)
    dist = e / np.sum(e)
    return dist

def SDAE_encode(model_name,X_c,layer_structure, W, b,batch_normalization, f_act, model_keep_prob,V_u=None):
    hidden_value=X_c
    for iter in range(int(len(layer_structure)-2)):#说明是encode阶段
        if V_u!=None:
            before_activation = tf.add(tf.add(tf.matmul(hidden_value, W[iter]), V_u), b[iter])
        else:
            before_activation = tf.add(tf.matmul(hidden_value, W[iter]), b[iter])
        #在最后的隐含值输出时需要分成两块，均值以及标准差
        if batch_normalization == "True":
            before_activation = batch_norm(before_activation)
        hidden_value = f_act(before_activation)

        if iter< len(layer_structure):
            hidden_value = tf.nn.dropout(hidden_value, model_keep_prob)

    gaussian_params = tf.matmul(hidden_value, W[int(len(layer_structure))-2]) + b[int(len(layer_structure))-2]
    mean=gaussian_params[:, :int(layer_structure[int(len(layer_structure))-1]/2)]#main函数中设定的隐藏神经元的个数==10
    stddev = 1e-6 + tf.nn.softplus(gaussian_params[:, int(layer_structure[int(len(layer_structure))-1]/2):])#隐藏神经元
    # sampling by re-parameterization technique
    Encoded_X = mean + stddev* tf.random_normal(tf.shape(mean), 0, 1, dtype=tf.float32)

    return  mean,stddev,Encoded_X


def SDAE_decode(model_name,Encoded_X,layer_num,layer_structure, W, b,batch_normalization, g_act, model_keep_prob):
    hidden_value=Encoded_X
    for iter in range(len(layer_structure)-2):
        before_activation = tf.add(tf.matmul(hidden_value, W[iter+int(layer_num/2)]), b[iter+int(layer_num/2)])
        if batch_normalization == "True":
            before_activation = batch_norm(before_activation)
        if iter< len(layer_structure)-1:
            hidden_value = tf.nn.dropout(hidden_value, model_keep_prob)

        hidden_value = g_act(before_activation)

    sdae_output = tf.sigmoid(tf.matmul(hidden_value, W[int(len(layer_structure)+int(layer_num/2)-2)]) +b[int(len(layer_structure)+int(layer_num/2)-2)])

    return sdae_output
