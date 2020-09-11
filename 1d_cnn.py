import os
import csv
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn import metrics
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
from keras.models import Sequential
from keras.regularizers import l2
from keras.layers import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers import  Flatten
from keras.layers import Conv1D, MaxPooling1D,Reshape,GlobalAveragePooling1D

parentDir = os.getcwd()
with open (parentDir+'\\'+'imbalance2_denoised.csv','r')as f:#PCA2_7，imbalance2
    data_2=pd.read_csv(f,header=0,index_col=0,encoding='utf_8')

##2号训练集
H_start_num=131104
H_end_num=164889
health_num=H_end_num-H_start_num

F_start_num=164890
F_end_num=len(data_2)
fault_num=F_end_num-F_start_num

X_training_data=np.vstack((data_2.iloc[H_start_num:H_end_num],data_2.iloc[F_start_num:F_end_num]))
y_train = np.array([0]*health_num+[1]*fault_num)
permutation_train = np.random.permutation(y_train.shape[0])
x_train = X_training_data[permutation_train, :]
y_train = y_train[permutation_train]
print(x_train.shape)
print(y_train.shape)

with open (parentDir+'\\'+'imbalance14_denoised.csv','r')as f:#PCA14_7，imbalance14
    data_14=pd.read_csv(f,header=0,index_col=0,encoding='utf_')

##14号验证集
H_start_num=1
H_end_num=55337
health_num=H_end_num-H_start_num

F_start_num=55358
F_end_num=len(data_14)
fault_num=F_end_num-F_start_num

X_testing_data=np.vstack((data_14.iloc[H_start_num:H_end_num],data_14.iloc[F_start_num:F_end_num]))
y_test = np.array([0]*health_num+[1]*fault_num)
permutation_test = np.random.permutation(y_test.shape[0])

x_test = X_testing_data[permutation_test, :]
y_test = y_test[permutation_test]
print(x_test.shape)
print(y_test.shape)
standard_scaler=preprocessing.StandardScaler()
x_train=standard_scaler.fit_transform(x_train)
x_test=standard_scaler.fit_transform(x_test)

def train_summary(history):
    print("\n--- Learning curve of model training ---")
    plt.figure(figsize=(10, 8))
    sns.set_style("whitegrid")
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['font.family'] = 'sans-serif'
    plt.plot(history.history['accuracy'], "go-", label="训练正确率")
    plt.plot(history.history['val_accuracy'], "g", label="验证正确率")
    plt.plot(history.history['loss'], "ro-", label="训练误差")
    plt.plot(history.history['val_loss'], "r", label="验证误差")
    plt.title('模型正确率及误差',fontsize = 15)
    plt.ylabel('正确率及误差',fontsize = 15)
    plt.xlabel('迭代次数',fontsize = 15)
    plt.ylim(0)
    plt.legend()
    plt.show()

#混淆矩阵图
def show_confusion_matrix(validations, predictions):
    LABELS = ['healthy','fault']
    matrix = metrics.confusion_matrix(validations, predictions)
    plt.figure(figsize=(8, 8))
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['font.family'] = 'sans-serif'
    sns.heatmap(matrix,cmap="coolwarm",linecolor='white',
                linewidths=1,xticklabels=LABELS,yticklabels=LABELS,
                annot=True,fmt="d",annot_kws={'size':18})
    plt.title("混淆矩阵",fontsize = 15)
    plt.ylabel("真实类别",fontsize = 15)
    plt.xlabel("预测类别",fontsize = 15)
    plt.show()

def Model_train(model,x_train,y_train,batch_size,epochs):
    print("\n--- Fit the model ---\n")
#     callbacks_list = [keras.callbacks.ModelCheckpoint(
#                       filepath='model/my_model_raw_1234.h123',
#                       monitor='val_loss', save_best_only=True),
#                       keras.callbacks.EarlyStopping(monitor='acc', patience=5)]
    history = model.fit(x_train,y_train,batch_size=batch_size,epochs=epochs,
                       validation_split=0.2)
    return history

def CNN_Model(maxlen,reg=l2(0.0005), init="he_normal"):
    #1维卷积神经网络
    model = Sequential()
    model.add(Reshape((maxlen, 1),input_shape=(maxlen,)))
    model.add(Conv1D(filters,
                 kernel_size,
                 padding='valid',#多的dropped，不会补零
                 activation='relu',
                 input_shape=(maxlen, 1),
                 strides=1))
    #model.add(BatchNormalization())
    #model.add(Conv1D(32, 3, strides=2, padding="same", kernel_initializer=init,
                     #kernel_regularizer=reg,activation='relu'))
    #model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=pool_size))
    model.add(Flatten())
    model.add(Dropout(0.25))
    model.add(Dense(1, activation='sigmoid'))
    print(model.summary())
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
    return model

def Model_test(model,x_test,y_test):
    print("\n--- Check against test data ---")
    score = model.evaluate(x_test, y_test, verbose=1)
    print("\nAccuracy on test data: %0.2f" % score[1])
    print("\nLoss on test data: %0.2f" % score[0])

    y_pred_test = model.predict(x_test)
    max_y_pred_test = np.argmax(y_pred_test, axis=1)
    max_y_test = np.argmax(y_test, axis=1)
    show_confusion_matrix(max_y_test, max_y_pred_test)

    print("\n--- Classification report for test data ---")
    print(classification_report(max_y_test, max_y_pred_test))


if __name__ == '__main__':
    #1.训练
    maxlen = x_train.shape[1]
    kernel_size = 3
    filters = 32
    pool_size = 3
    # LSTM
    lstm_output_size = 100
    # Training
    batch_size = 100
    epochs = 20
    CNN_Model = CNN_Model(maxlen)
    history = Model_train(CNN_Model,x_train,y_train,batch_size,epochs)
    train_summary(history)
    score, acc =CNN_Model.evaluate(x_test, y_test, batch_size=batch_size)
    print("\n--- Check against test data ---")
    y_pred_test = CNN_Model.predict_classes(x_test)
    show_confusion_matrix(y_test,y_pred_test)
    print("\nAccuracy on test data: %0.2f" % acc)
    print("\nLoss on test data: %0.2f" %  score)