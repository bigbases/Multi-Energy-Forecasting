import pandas as pd
import numpy as np
import sys

from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error

ENERGY = sys.argv[1]


print('tape_Att_baseline_concat_FCL_{}'.format(ENERGY))
for num in range(131,136) :
    print('{}_running'.format(num))
    def preprocessing(data) :
        feature = data.copy()[['time','elec','water','gas','hotwater','hot']]
        feature.time = pd.to_datetime(feature.time)
        feature.set_index('time',inplace=True)
        y = data[[ENERGY]]
        return feature, y

    train = pd.read_csv('test1month_train_2_summer.csv')
    # train = pd.read_csv('test1month_train_summer.csv')
    valid = pd.read_csv('test1month_valid_summer.csv')
    test = pd.read_csv('test1month_test_summer.csv')

    X_train, y_train = preprocessing(train)
    X_valid, y_valid = preprocessing(valid)
    X_test, y_test = preprocessing(test)


    scaler = MinMaxScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train))
    X_valid = pd.DataFrame(scaler.transform(X_valid))
    X_test = pd.DataFrame(scaler.transform(X_test))



    def timeseries_data(dataset, target, start_index, end_index, window_size, target_size) :
        data = []
        labels = []

        y_start_index = start_index + window_size # 0+24
        y_end_index = end_index - target_size  # train_index(10291) - 24 = 10267

        for i in range(y_start_index, y_end_index) :
            data.append(dataset.iloc[i-window_size:i,:].values)
            labels.append(target.iloc[i:i+target_size,:].values)
        data = np.array(data)
        labels = np.array(labels)
        labels = labels.reshape(-1,target_size)  
        return data, labels

    # making window
    window = 72
    X_train, y_train = timeseries_data(X_train,y_train,0,len(X_train),window,24)
    X_valid, y_valid = timeseries_data(X_valid,y_valid,0,len(X_valid),window,24)
    X_test, y_test = timeseries_data(X_test,y_test,0,len(X_test),window,24)

    batch_size = 64
    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(X_train.shape[0],seed=42).batch(batch_size, drop_remainder=True).prefetch(1)
    valid_ds = tf.data.Dataset.from_tensor_slices((X_valid, y_valid)).batch(batch_size, drop_remainder=True).prefetch(1)
    test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(batch_size, drop_remainder=True).prefetch(1)

    class Encoder(keras.layers.Layer) :
        def __init__(self, name="encoder"):
            super(Encoder, self).__init__(name=name)
            self.lstm2 = keras.layers.LSTM(256, return_sequences=True, kernel_initializer='he_normal')# unit, return_sequence 변경
            self.flatten = keras.layers.Flatten() # 추가
            self.output_ = keras.layers.Dense(24, kernel_initializer= 'he_normal')

        def call(self, x) :
            x = self.lstm2(x)
            x = self.flatten(x) #추가
            x = self.output_(x)
            return x
        
    class AttentionModel(keras.Model):

        def __init__(self,name="attentionmodel"):
            super(AttentionModel, self).__init__(name=name)
            self.encoder1 = Encoder()
            self.encoder2 = Encoder()
            self.encoder3 = Encoder()
            self.encoder4 = Encoder()
            self.encoder5 = Encoder()
            self.flatten = keras.layers.Flatten()
            self.hidden1 = keras.layers.Dense(512, kernel_initializer='he_normal',activation = 'relu')
            self.hidden2 = keras.layers.Dense(256, kernel_initializer= 'he_normal',activation = 'relu')
            self.output_ = keras.layers.Dense(24, kernel_initializer= 'he_normal')

        def call(self, input1, input2, input3, input4, input5):
            out1 = self.encoder1(tf.expand_dims(input1,2))
            out2 = self.encoder2(tf.expand_dims(input2,2))
            out3 = self.encoder3(tf.expand_dims(input3,2))
            out4 = self.encoder4(tf.expand_dims(input4,2))
            out5 = self.encoder5(tf.expand_dims(input5,2))
            out1 = tf.expand_dims(out1,1)
            out2 = tf.expand_dims(out2,1)
            out3 = tf.expand_dims(out3,1)
            out4 = tf.expand_dims(out4,1)
            out5 = tf.expand_dims(out5,1)
            concat_energy = tf.concat([out1,out2,out3,out4,out5],axis=1)
            flatten = self.flatten(concat_energy)
            hidden1 = self.hidden1(flatten)
            hidden2 = self.hidden2(hidden1)
            output = self.output_(hidden2)

            return output

    class EarlyStopping:
        
        def __init__(self, patience=30, verbose=False, delta=0):
            self.patience = patience
            self.verbose = verbose
            self.counter = 0
            self.best_score = None
            self.early_stop = False
            self.val_loss_min = np.Inf
            self.delta = delta

        def __call__(self, val_loss, model):

            score = -val_loss

            if self.best_score is None:
                self.best_score = score
                self.save_checkpoint(val_loss, model)
            elif score < self.best_score + self.delta:
                self.counter += 1
                # print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                # if self.counter >= self.patience:
                #     self.early_stop = True
            else:
                self.best_score = score
                self.save_checkpoint(val_loss, model)
                self.counter = 0
            
            return self.early_stop

        def save_checkpoint(self, val_loss, model):
            '''Saves model when validation loss decrease.'''
            # if self.verbose:
            #     print(f'loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).')
            # print(f'loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).')
            model.save_weights('./checkpoints/my_checkpoint')
            self.val_loss_min = val_loss

    model = AttentionModel()
    n_epochs = 200
    # boundaries = [99]
    # values = [0.001, 0.0001]
    # learning_rate_fn = keras.optimizers.schedules.PiecewiseConstantDecay(boundaries, values)
    optimizer = keras.optimizers.Adam()
    loss_fn_1 = keras.losses.MeanSquaredError()
    loss_fn_test = keras.losses.MeanSquaredError()
    metric = keras.metrics.MeanAbsoluteError()
    early = EarlyStopping()


    # For Train set
    print('training start')
    out1_df = []
    concat_df = []
    result_df = pd.DataFrame({'MAPE' : [], 'MAE' : [], 'RMSE' : []})
    loss_df = []
    loss_test = []
    val_loss = []
    for epoch in range(n_epochs) :
        # lr = learning_rate_fn(epoch)
        # optimizer.learning_rate = lr
        loss_batch = 0
        for batch,(features, label) in train_ds.enumerate() :
            with tf.GradientTape() as tape :
                y_pred = model(features[:,:,0],features[:,:,1],features[:,:,2],
                                                         features[:,:,3], features[:,:,4], training=True)
                loss = loss_fn_1(label, y_pred)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            loss_batch += loss
        loss_df.append(loss_batch)
        # For valid set
        sk_metric = 0
        for batch,(features, label) in valid_ds.enumerate() :
            y_pred = model(features[:,:,0],features[:,:,1],features[:,:,2],features[:,:,3],features[:,:,4], training=False)
            error_mae = mean_absolute_error(label, y_pred)
            sk_metric+=error_mae
        # print(val_metric)
        # print('sk',sk_metric,'epoch',epoch)
        val_loss.append(sk_metric)
        if early(sk_metric, model):
            print(early(sk_metric, model))
            break

    model.load_weights('./checkpoints/my_checkpoint')
    predict_df = []
    real_df = []
    loss_ = 0
    for batch,(features, label) in test_ds.enumerate() :
        y_pred = model(features[:,:,0],features[:,:,1],features[:,:,2],features[:,:,3],features[:,:,4], training=False)
        loss_+=loss_fn_test(label, y_pred)
        predict_df.append(y_pred)
        real_df.append(label)
    loss_test.append(loss_)

    real_new = [a.numpy() for b in real_df for a in b]
    predict_new = [a.numpy().round(1) for b in predict_df for a in b] 


    error_mape = mean_absolute_percentage_error(real_new, predict_new)
    error_mae = mean_absolute_error(real_new, predict_new)
    error_rmse = mean_squared_error(real_new, predict_new)**(0.5)
    result = pd.DataFrame({'MAPE' : [error_mape], 'MAE' : [error_mae], 'RMSE' : [error_rmse]})
    result_df = pd.concat([result_df,result],axis=0)
    print(result_df)
    
    result_df.to_pickle('./base_lstm1_{}/result_{}.csv'.format(ENERGY, num))
    loss_df = [i.numpy() for i in loss_df]
    pd.DataFrame(loss_df).to_pickle('./base_lstm1_{}/loss_{}.csv'.format(ENERGY, num))
    loss_test = [i.numpy() for i in loss_test]
    pd.DataFrame(loss_test).to_pickle('./base_lstm1_{}/loss_test_{}.csv'.format(ENERGY, num))
    # val_loss = [i for i in val_loss]
    pd.DataFrame(val_loss).to_pickle('./base_lstm1_{}/loss_val_{}.csv'.format(ENERGY, num))
