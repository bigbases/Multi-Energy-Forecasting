import pandas as pd
import numpy as np
import sys

from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error
import tensorflow.keras.backend as K # 모델에 있는 func 추가
from tensorflow.keras import Model # 모델에 있는 func 추가 

ENERGY = sys.argv[1]


print('tape_Att_bahdanau_sigmoid_concat_FCL_{}'.format(ENERGY))
for num in range(100,105) :
    print('{}_running'.format(num))
    def preprocessing(data) :
        if ENERGY == 'elec' :
            feature = data.copy()[['time','elec','water','gas','hotwater','hot']]
        elif ENERGY == 'water' :
            feature = data.copy()[['time','water','elec','gas','hotwater','hot']]
        elif ENERGY == 'gas' :
            feature = data.copy()[['time','gas','elec','water','hotwater','hot']]
        elif ENERGY == 'hotwater' :
            feature = data.copy()[['time','hotwater','elec','water','gas','hot']]
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

        y_start_index = start_index + window_size
        y_end_index = end_index - target_size

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

    batch_size = 32
    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(X_train.shape[0],seed=42).batch(batch_size, drop_remainder=True).prefetch(1)
    valid_ds = tf.data.Dataset.from_tensor_slices((X_valid, y_valid)).batch(batch_size, drop_remainder=True).prefetch(1)
    test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(batch_size, drop_remainder=True).prefetch(1)

    class GenericBlock(keras.layers.Layer):
        def __init__(self, lookback, horizon, num_neurons, block_layers):
            super(GenericBlock, self).__init__()
            # collection of layers in the block    
            self.layers_       = [keras.layers.Dense(num_neurons, activation = 'relu') 
                                  for _ in range(block_layers)]
            self.lookback      = lookback
            self.horizon       = horizon

            # multiply lookback * forecast to get training window size
            self.backcast_size = lookback

            # numer of neurons to use for theta layer -- this layer
            # provides values to use for backcast + forecast in subsequent layers
            self.theta_size    = self.backcast_size + horizon

            # layer to connect to Dense layers at the end of the generic block
            self.theta         = keras.layers.Dense(self.theta_size, 
                                                    use_bias = False, 
                                                    activation = None)

        def call(self, inputs):
            # save the inputs
            x = inputs
            # connect each Dense layer to itself
            for layer in self.layers_:
                x = layer(x)
            # connect to Theta layer
            x = self.theta(x)
            # return backcast + forecast without any modifications
            return x[:, :self.backcast_size], x[:, -self.horizon:]

    ### CREATES NESTED LAYERS INTO A SINGLE NBEATS LAYER
    class NBeats(keras.layers.Layer):
        def __init__(self,
                     model_type           = 'generic',
                     lookback             = 72, 
                     horizon              = 24, 
                     num_generic_neurons  = 512,
                     num_generic_stacks   = 10,
                     num_generic_layers   = 4,
                     num_trend_neurons    = 256,
                     num_trend_stacks     = 3,
                     num_trend_layers     = 4,
                     num_seasonal_neurons = 2048,
                     num_seasonal_stacks  = 3,
                     num_seasonal_layers  = 4,
                     num_harmonics        = 1,
                     polynomial_term      = 3,
                     **kwargs):
            super(NBeats, self).__init__()
            self.model_type           = model_type
            self.lookback             = lookback
            self.horizon              = horizon
            self.num_generic_neurons  = num_generic_neurons
            self.num_generic_stacks   = num_generic_stacks
            self.num_generic_layers   = num_generic_layers
            self.num_trend_neurons    = num_trend_neurons
            self.num_trend_stacks     = num_trend_stacks
            self.num_trend_layers     = num_trend_layers
            self.num_seasonal_neurons = num_seasonal_neurons
            self.num_seasonal_stacks  = num_seasonal_stacks
            self.num_seasonal_layers  = num_seasonal_layers
            self.num_harmonics        = num_harmonics
            self.polynomial_term      = polynomial_term

            # Model architecture is pretty simple: depending on model type, stack
            # appropriate number of blocks on top of one another
            # default values set from page 26, Table 18 from paper
            if model_type == 'generic':
                self.blocks_ = [GenericBlock(lookback       = lookback, 
                                             horizon        = horizon,
                                             num_neurons    = num_generic_neurons, 
                                             block_layers   = num_generic_layers)
                                 for _ in range(num_generic_stacks)]

        def call(self, inputs):
            residuals = K.reverse(inputs, axes = 0)
            forecast  = inputs[:, -1:]
            for block in self.blocks_:
                backcast, block_forecast = block(residuals)
                residuals = keras.layers.Subtract()([residuals, backcast])
                forecast  = keras.layers.Add()([forecast, block_forecast])
            return forecast
        
    class Corr(keras.layers.Layer):
        def __init__(self, name="corr"):
            super(Corr, self).__init__(name=name)
            self.W1 = tf.keras.layers.Dense(24, kernel_initializer= 'he_normal')
            self.W2 = tf.keras.layers.Dense(24, kernel_initializer= 'he_normal')
            self.V = keras.layers.Dense(1, kernel_initializer= 'he_normal')

        def call(self, out1, out2, out3, out4, out5) :
            concat_energy = tf.concat([out2,out3,out4,out5],axis=1)
            bahdanau = tf.nn.tanh(self.W1(out1) + self.W2(concat_energy))
            score = self.V(bahdanau)
            weight = keras.activations.sigmoid(tf.squeeze(score))

            return weight

    class AttentionModel(keras.Model):
        def __init__(self, name="attentionmodel"):
            super(AttentionModel, self).__init__(name=name)
            self.encoder1 = NBeats()
            self.encoder2 = NBeats()
            self.encoder3 = NBeats()
            self.encoder4 = NBeats()
            self.encoder5 = NBeats()
            self.corr = Corr()
            self.flatten = keras.layers.Flatten()
            self.hidden1 = keras.layers.Dense(512, kernel_initializer='he_normal',activation = 'relu')
            self.hidden2 = keras.layers.Dense(256, kernel_initializer= 'he_normal',activation = 'relu')
            self.output_ = keras.layers.Dense(24, kernel_initializer= 'he_normal')

        def call(self, input1, input2, input3, input4, input5):
            out1 = self.encoder1(input1)
            out2 = self.encoder2(input2)
            out3 = self.encoder3(input3)
            out4 = self.encoder4(input4)
            out5 = self.encoder5(input5)
            out1 = tf.expand_dims(out1,1)
            out2 = tf.expand_dims(out2,1)
            out3 = tf.expand_dims(out3,1)
            out4 = tf.expand_dims(out4,1)
            out5 = tf.expand_dims(out5,1)
            weight = self.corr(out1, out2, out3, out4, out5)
            concat_energy = tf.concat([out2,out3,out4,out5],axis=1)
            context_vector = tf.expand_dims(weight,2)*concat_energy
            concat_att_elec = tf.concat([context_vector, out1], axis=1)
            flatten = self.flatten(concat_att_elec)
            hidden1 = self.hidden1(flatten)
            hidden2 = self.hidden2(hidden1)
            output = self.output_(hidden2)

            return output, weight
        
    class EarlyStopping:
        """Early stops the training if validation loss doesn't improve after a given patience."""
        def __init__(self, patience=20, verbose=False, delta=0):
            """
            Args:
                patience (int): How long to wait after last time validation loss improved.
                                Default: 7
                verbose (bool): If True, prints a message for each validation loss improvement. 
                                Default: False
                delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                                Default: 0
            """
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
                # print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
            print(f'loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).')
            model.save_weights('./checkpoints/my_checkpoint')
            self.val_loss_min = val_loss

    model = AttentionModel()

    n_epochs = 300
    # boundaries = [99]
    # values = [0.001, 0.0001]
    # learning_rate_fn = keras.optimizers.schedules.PiecewiseConstantDecay(boundaries, values)
    optimizer = keras.optimizers.Adam()
    loss_fn_1 = keras.losses.MeanAbsoluteError()
    loss_fn_test = keras.losses.MeanAbsoluteError()
    early = EarlyStopping()


    # For Train set
    print('training start')
    result_df = pd.DataFrame({'MAPE' : [], 'MAE' : [], 'RMSE' : []})
    loss_df = []
    loss_test = []
    val_loss = []
    for epoch in range(n_epochs) :
        for batch,(features, label) in train_ds.enumerate() :
            # lr = learning_rate_fn(epoch)
            # optimizer.learning_rate = lr
            loss_batch = 0
            with tf.GradientTape() as tape :
                y_pred, weight = model(features[:,:,0],features[:,:,1],features[:,:,2],features[:,:,3], features[:,:,4], training=True)
                loss = loss_fn_1(label, y_pred)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            loss_batch += loss
            if epoch == 199 :
                corr_list = np.zeros((32,4))
                for i in range(features.shape[0]) : 
                    [[corr_main, corr_sub1, corr_sub2, corr_sub3, corr_sub4]]= pd.DataFrame(np.stack(list(features))[i]).corr().iloc[0:1,:].values
                    if np.isnan(corr_sub4) == True :
                        corr_sub4 = 0
                    corr_list[i,0], corr_list[i,1], corr_list[i,2], corr_list[i,3] = corr_sub1, corr_sub2, corr_sub3, corr_sub4
        loss_df.append(loss_batch)
        
        # For valid set
        sk_metric = 0
        for batch,(features, label) in valid_ds.enumerate() :
            y_pred, _ = model(features[:,:,0],features[:,:,1],features[:,:,2],features[:,:,3],features[:,:,4], training=False)
            error_mae = mean_absolute_error(label, y_pred)
            sk_metric+=error_mae
        print('sk',sk_metric,'epoch',epoch)
        val_loss.append(sk_metric)
        if early(sk_metric, model):
            print(early(sk_metric, model))
            break
            
            
    model.load_weights('./checkpoints/my_checkpoint')
    predict_df = []
    real_df = []
    loss_ = 0
    for batch,(features, label) in test_ds.enumerate() :
        y_pred, _ = model(features[:,:,0],features[:,:,1],features[:,:,2],features[:,:,3],features[:,:,4], training=False)
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
    print(result)

    result_df.to_pickle('./bahda_{}/result_{}.csv'.format(ENERGY, num))
    loss_df = [i.numpy() for i in loss_df]
    pd.DataFrame(loss_df).to_pickle('./bahda_{}/loss_{}.csv'.format(ENERGY, num))
    loss_test = [i.numpy() for i in loss_test]
    pd.DataFrame(loss_test).to_pickle('./bahda_{}/loss_test_{}.csv'.format(ENERGY, num))
    pd.DataFrame(weight.numpy()).to_pickle('./bahda_{}/weight_{}.csv'.format(ENERGY, num))
    pd.DataFrame(corr_list).to_pickle('./bahda_{}/corrlist_{}.csv'.format(ENERGY, num))
    pd.DataFrame(val_loss).to_pickle('./bahda_{}/loss_val_{}.csv'.format(ENERGY, num))
