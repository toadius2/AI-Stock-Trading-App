import threading
import time
import pickle
import matplotlib.pyplot as plt
import datetime
import datetime as dt
import ccxt
import pandas as pd
pd.core.common.is_list_like = pd.api.types.is_list_like
from pandas_datareader import data
import numpy as np
import operator
import Robinhood as rh
import requests
import json
import csv
import os
import os.path
import matplotlib.pyplot as plt
import urllib.request, json 
import tensorflow as tf
import GUI_Robinhood

#Misc Imports 
from bs4 import BeautifulSoup
from newspaper import Article
from nltk.tokenize import sent_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from collections import defaultdict


#Keras Import
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder, MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from operator import itemgetter
from keras import Sequential
from keras.layers import Dense, LSTM, Masking
from sklearn.utils import compute_class_weight, compute_sample_weight
from keras.models import load_model
from keras.callbacks import Callback
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping, ModelCheckpoint
#Kivy importsd 
from kivy.properties import NumericProperty
from kivy.properties import StringProperty
from kivy.uix.scatter import Scatter
from kivy.app import App
from kivy.lang import Builder
from kivy.factory import Factory
from kivy.animation import Animation
from kivy.clock import Clock, mainthread
from kivy.uix.gridlayout import GridLayout
from collections import defaultdict
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.properties import ObjectProperty

class PerfectTrade(Screen):

    stop = threading.Event()
    
    total_calls = 0
    successes = 0

    def run(self):
        threading.Thread(target=self.second_thread).start()

    def second_thread(self):
        Clock.schedule_once(self.start_test, 0)

        time.sleep(5)

        time.sleep(2)

        self.stop_test()

        threading.Thread(target=self.run_final_script(6459.2, 1541352660000)).start()

    def start_test(self, *args):
        self.remove_widget(self.start_perfect_trade_button)
        self.after_run_label.text = ('Please wait...')
        anim_bar = Factory.AnimWidget()
        self.anim_box.add_widget(anim_bar)
        anim = Animation(opacity=0.3, width=100, duration=0.6)
        anim += Animation(opacity=1, width=400, duration=0.8)
        anim.repeat = True
        anim.start(anim_bar)

    @mainthread
    def update_label_main(self, new_text):
        self.pre_run_label.text = new_text
    @mainthread
    def update_label_status(self, new_text):
        self.status.text = new_text

    @mainthread
    def stop_test(self):
        self.after_run_label.text = ('Currently running Perfect Trade!')

        self.remove_widget(self.anim_box)


    def run_final_script(self, triggerprice, starttime_ms):
        # main method to load saved model and update iteratively with new data
        self.model = load_model('best_model.h5')
        self.update_label_main("Fetching Latest Data...")
        self.from_datetime = str(datetime.datetime.fromtimestamp(starttime_ms/1000.0))
        self.symbol = 'BTC/USD'
        self.binance = ccxt.binance({'rateLimit': 20, 'enableRateLimit': True})
        self.new_data = self.print_chart(self.binance, 'BTC/USDT', '1m', self.from_datetime, fetch_status=False)
       
        # format initial data
        self.trainX, self.train_target_vec, self.newTriggerPrice = self.batch_format(self.new_data, triggerprice, 103)
        self.last_timestamp = self.new_data['Timestamp_ms'].iloc[-1]
        del self.train_target_vec[-1]

        # set up model
        self.model_reInit(self.trainX, self.train_target_vec, self.model)
        self.update_label_main("Beginning Real Time Training and Prediction")
        self.t1 = time.time()
        model_generated_data = []
        
        # set up the graph
        self.buy_sell_graph_setup()
        x = list()
        y = list()
        x_valley = list()
        y_valley = list()
        x_peak = list()
        y_peak = list()
        
        thirty_minute_mode = False # TODO: Allow the user to switch between the 2 graphs
        x_current = list()
        y_current = list()
        x_valley_30 = list()
        y_valley_30 = list()
        x_peak_30 = list()
        y_peak_30 = list()

        try:
            while True:

                self.from_datetime = str(datetime.datetime.fromtimestamp(self.last_timestamp / 1000.0))
                self.new_data = self.print_chart(self.binance, 'BTC/USDT', '1m', self.from_datetime, fetch_status=False)
                self.time_diff = self.new_data['Timestamp_ms'].iloc[-1] - self.last_timestamp
                
                #Real Data For writing to CSV from Binance
                self.labeled_prices, _ = self.label_perfect_trades(self.new_data, triggerprice) 
                self.labeled_prices.to_csv('real_labeled_price_data.csv')


                self.t2 = time.time()
                self.curr_time_diff = self.t2-self.t1
                time.sleep(0.5)

                if ( self.time_diff >= 60000):
                    self.last_timestamp = self.new_data['Timestamp_ms'].iloc[-1]
                    self.pred_time =  datetime.datetime.fromtimestamp(self.last_timestamp / 1000.0)
                    self.last_datapoint = self.new_data.tail(2)
                    self.trainX, _ , _= self.batch_format(self.last_datapoint, self.newTriggerPrice, 103)
                    pred_time1 = str(self.pred_time)
                    self.theText =  str("Prediction for BTC Data at "+ pred_time1)
                    self.update_label_main(self.theText)
                    print("###########THIS IS THE MODEL################")
                    print(self.model_predict(self.trainX, self.model)[0])
                    
                    #Appends the timestamp and the generated Artificial qualification to a list. 
                    model_prediction = self.model_predict(self.trainX, self.model)[0]
                    model_generated_data.append([self.last_timestamp, model_prediction])
                    print("\n",self.model_predict(self.trainX, self.model))
                    
                    #Changes aartificial data from a list of lists, to a dataframe for 
                    self.prediction_dataframe = pd.DataFrame(model_generated_data, columns=['Timestamp_ms','Qualifier'])
                    #Writes Artificial data to a CSV
                    self.prediction_dataframe.to_csv('artificial_labeled_data.csv')
                    
                    
                    #total_calls, successes, _ += self.test_artificial_data(self.prediction_dataframe,self.labeled_prices)
                    success_rate = self.test_artificial_data(self.prediction_dataframe,self.labeled_prices)

                    print(self.total_calls, self.successes, success_rate)
                    print (self.prediction_dataframe)
                
                    x.append(self.pred_time)
                    x_current.append(self.pred_time)
                    actual_last_datapoint = self.new_data.tail(1)
                    y_data = actual_last_datapoint['Closing_Price'].tolist()
                    y.append(y_data)
                    y_current.append(y_data)
        
                    if thirty_minute_mode == True:
        
                        if len(x_current) >= 31:
                            if len(x_valley_30) >=1:
                                if x_current[0] == x_valley_30[0]:
                                    x_valley_30.pop(0)
                                    y_valley_30.pop(0)
                            if len(x_peak_30) >=1:
                                if x_current[0] == x_peak_30[0]:
                                    x_peak_30.pop(0)
                                    y_peak_30.pop(0)
                            x_current.pop(0)
                            y_current.pop(0)
        
                        self.buy_sell_graph(y_data,x_current,y_current,x_valley_30,y_valley_30,x_peak_30,y_peak_30,model_prediction,self.pred_time)
                    else:
                        self.buy_sell_graph(y_data,x,y,x_valley,y_valley,x_peak,y_peak,model_prediction,self.pred_time)
                    
                    time.sleep(61)  
                    self.data_update  = self.print_chart(self.binance, 'BTC/USDT', '1m', self.from_datetime, fetch_status=False)
                    
                    if (self.data_update['Timestamp_ms'].iloc[-1] - self.last_timestamp >= 60000):
                        
                        self.last_timestamp = self.data_update['Timestamp_ms'].iloc[-2]
                        self.pred_time = str(datetime.datetime.fromtimestamp(self.last_timestamp / 1000.0))
                        self.theText = str("True Label for BTC Data at " + self.pred_time)
                        self.update_label_main(self.theText)
                        self.trainX, self.train_target_vec , self.newTriggerPrice= self.batch_format(self.last_datapoint, self.newTriggerPrice, 103)
                        self.trainX[0][102] =  self.trainX[0][101]
                        self.trainX[0][101] = [0.0,0.0,0.0,0.0,0.0]
                        
                        #self.print_format_labels(self.trainX)
                        #self.model_update(self.trainX, self.train_target_vec, self.model)

                        if len(self.trainX) == len(self.train_target_vec):
                            self.model_update(self.trainX, self.train_target_vec, self.model)
                    
                    self.t1 = time.time()

        except KeyboardInterrupt:
            pass

    class epoch_update(Callback):
        def on_train_begin(self, logs={}):
            self.losses = []

        def on_batch_end(self, batch, logs={}):
            self.losses.append(logs.get('loss'))
            if len(self.losses) % 10 == 0:
                print("Completed ", len(self.losses), " of ", " 50 epochs")


    def plot_confusion_matrix(self, cm, classes,
                              normalize=False,
                              title='Confusion matrix',
                              cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            self.update_label_main("Normalized confusion matrix")
        else:
            print("Confusion matrix, without normalization")

        print(cm)

        plt.imshow(cm, self.interpolation=='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        self.tick_marks = np.arange(len(classes))
        plt.xticks(self.tick_marks, classes, rotation=45)
        plt.yticks(self.tick_marks, classes)

        self.fmt = '.2f' if normalize else 'd'
        self.thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()


    def val_loss_graph(self, model_history):
        # val loss and train loss graph
        print(model_history.history.keys())
        plt.plot(model_history.history['loss'])
        plt.plot(model_history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()


    def build_model(self, train_data, train_targets, all_targets_for_weight):
        # used to build initial model
        # saves best model
        #returns model

        # Compute class weight (v imbalanced)
        self.y_classes = all_targets_for_weight

        # Instantiate the label encoder
        self.le = LabelEncoder() #this encodes 

        # Fit the label encoder to our label series
        self.le.fit(list(self.y_classes))

        # Create integer based labels Series
        self.y_integers = self.le.transform(list(self.y_classes))

        # Create dict of labels : integer representation
        self.labels_and_integers = dict(zip(self.y_classes, self.y_integers))
        self.class_weights = compute_class_weight('balanced', np.unique(self.y_integers), self.y_integers)
        self.sample_weights = compute_sample_weight('balanced', self.y_integers)

        self.class_weights_dict = dict(zip(self.le.transform(list(self.le.classes_)), self.class_weights))

        # In[15]:

        # KERAS LSTM MODEL
        self.model = Sequential()
        self.model.add(Masking(mask_value=0.0))

        self.model.add(LSTM(60, kernel_initializer='random_uniform'))
        # model.add(Dense(5, activation='relu' , kernel_initializer = 'random_uniform'))
        # model.add(Dropout(0.2))
        # model.add(Dense(3, activation='relu' , kernel_initializer = 'random_uniform'))
        self.model.add(Dense(3, activation='softmax'))
        self.model.compile(loss='mean_squared_error', optimizer='adam')

        self.callbacks = [ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)]
        history = self.model.fit(self.train_data, np.asarray(self.train_targets), epochs=500, validation_split=0.2, batch_size=10000, class_weight=self.class_weights_dict, verbose=1, shuffle=False, callbacks=self.callbacks)

        return history


    def model_predict(self, test_data, test_targets, model):
        self.test_target_condensed = []

        for x in test_targets:
            if x == [1, 0, 0]:
                self.test_target_condensed.append(0)
            if x == [0, 1, 0]:
                self.test_target_condensed.append(1)
            if x == [0, 0, 1]:
                self.test_target_condensed.append(2)

        # In[18]:

        # predict on test cases

        self.y_pred = model.predict_on_batch(self.test_data)
        self.y_classes = []
        for x in self.y_pred:
            self.index, self.value = max(enumerate(x), key=operator.itemgetter(1))
            self.y_classes.append(self.index)


    def model_predict(self, datapoint, model):
        self.y_pred = model.predict_on_batch(datapoint)
        self.test_target_condensed = []
        self.y_classes = []

        for x in self.y_pred:
            self.index, self.value = max(enumerate(x), key=operator.itemgetter(1))
            print("This is the self.index value for generated data.", self.index)
            print("This is the self.value value for generated data.", self.value)
            self.y_classes.append(self.index)

        for x in self.y_classes:
            if x == 0:
                self.test_target_condensed.append('neutral')
                self.update_label_status("Neutral")
            if x ==  1:
                self.test_target_condensed.append('valley')
                self.update_label_status("Valley")
            if x == 2:
                self.test_target_condensed.append('peak')
                self.update_label_status("Peak")

        return self.test_target_condensed


    def model_reInit(self, train_data, targets, model):
        self.out_batch = self.epoch_update()
        self.callbacks = [ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True), self.out_batch]
        self.update_label_main("Updating Model to Current Timestamp...")
        self.model.fit(train_data, np.asarray(targets), epochs=50, validation_split=0.2, verbose=0, batch_size=10000, shuffle=False, callbacks=self.callbacks)


    def model_update(self, train_data, targets, model):
        self.out_batch = self.epoch_update()
        self.callbacks = [self.out_batch]
        self.update_label_main("Updating Model to Current Timestamp...")
        self.model.fit(train_data, np.asarray(targets),
                  epochs=50, validation_split=0.2, verbose=0,
                  batch_size=10000,
                  shuffle=False, callbacks=self.callbacks)


    def print_format_labels(self, targets):
        self.index, self.value = max(enumerate(targets), key=operator.itemgetter(1))
        if self.index == 0:
            self.update_label_main("NEUTRAL")
        elif self.index == 1:
            self.update_label_main("VALLEY")
        elif self.index == 2:
            self.update_label_main("PEAK")

    def print_chart(self, exchange, symbol, timeframe, from_datetime, fetch_status=True):
        # function to get candle chart for time period
        # returns dataframe

        # PARAMS
        # exchange - ccxt exchange object that determines which exchange to pull OHLCVS from
        # symbol - string that represent symbol from respective exchange
        # timeframe - string for intervals between each price
        # from_datetime - when to begin price /ticker information

        self.msec = 1000
        self.minute = 60 * self.msec
        self.hour = 60
        self.hold = 30

        self.from_timestamp = exchange.parse8601(from_datetime)

        self.now = exchange.milliseconds()

        self.data = []

        while self.from_timestamp < self.now:

            try:
                if (fetch_status == True):
                    print(exchange.milliseconds(), 'Fetching candles starting from', exchange.iso8601(self.from_timestamp))
                self.ohlcvs = exchange.fetch_ohlcv(symbol, timeframe, self.from_timestamp)
                if (fetch_status == True):
                    print(exchange.milliseconds(), 'Fetched', len(self.ohlcvs), 'candles')
                if(len(self.ohlcvs)>1):
                    self.first = self.ohlcvs[0][0]
                    self.last = self.ohlcvs[-1][0]
                else:
                    self.first = self.ohlcvs
                    self.last = self.ohlcvs

                if (fetch_status == True):
                    print('First candle epoch', self.first, exchange.iso8601(self.first))
                    print('Last candle epoch', self.last, exchange.iso8601(self.last))


                # either remove or include additoinal params like day or month , whatever ccxt allows
                self.from_timestamp += len(self.ohlcvs) * self.minute
                self.data += self.ohlcvs

            except (ccxt.ExchangeError, ccxt.AuthenticationError, ccxt.ExchangeNotAvailable, ccxt.RequestTimeout) as error:

                print('Got an error', type(error).__name__, error.args, ', retrying in', self.hold, 'seconds...')
                time.sleep(self.hold)
        self.data = pd.DataFrame(self.data, columns=['Timestamp_ms', 'Open_Price', 'Highest_Price', 'Lowest_Price', 'Closing_Price',
                                           'Volume'])
        
        return self.data


    def label_perfect_trades(self, priceHistory, triggerPrice, new_training=False):
        # labels perfect trades as peaks , valleys or neutral

        # params
        # priceHistory - dataframe of OHLCVS
        self.label_df = pd.DataFrame(columns=['target'])
        
        self.label_df['Closing_Price'] = priceHistory['Closing_Price'].values
        if (new_training==True):
            triggerPrice = self.label_df['Closing_Price'].at[0]
        else:
            triggerPrice = triggerPrice
        for i in range(len(self.label_df['Closing_Price'].values) - 1):
            if self.label_df['Closing_Price'].at[i + 1] > triggerPrice + (triggerPrice * 0.001):
                self.label_df['target'].at[i + 1] = 2
                triggerPrice = self.label_df['Closing_Price'].at[i + 1]
            elif self.label_df['Closing_Price'].at[i + 1] < triggerPrice - (triggerPrice * 0.001):
                self.label_df['target'].at[i + 1] = 1

                triggerPrice = self.label_df['Closing_Price'].at[i + 1]
            else:
                self.label_df['target'].at[i + 1] = 0

        priceHistory = pd.concat([priceHistory, self.label_df.drop('Closing_Price', 1)], 1)
#        if (len(priceHistory) > 2):
#            priceHistory.fillna(0).to_csv('formatted_data.csv')

        return priceHistory.fillna(0), triggerPrice


    def formatAndSplit_data(self, data):

        ###
        # only really used for initial training/training large amount of data (i.e when training model from scratch)
        # formats data to timeseries representation
        # splits to train and test set
        # preprocesses data
        # groups for time series
        # returns trained set, train targets, test set, test_targets
        ###
        self.btc_df = data.drop(['target','Timestamp_ms'], 1)

        self.train_test_split = int(len(self.btc_df) * 0.8)

        data.drop('Timestamp_ms',1, inplace=True)

        self.targets = data['target']
        self.train_targets = self.targets[0:self.train_test_split].values
        self.test_targets = self.targets[self.train_test_split + 1:len(self.btc_df)].values
        self.btc_df = self.btc_df.values
        self.btc_train = self.btc_df[0:self.train_test_split, :]
        self.btc_test = self.btc_df[self.train_test_split + 1:len(self.btc_df), :]

        self.s = StandardScaler()
        self.train_s = self.s.fit_transform(self.btc_train)
        self.test_s = self.s.fit_transform(self.btc_test)

        # Format for time series
        # Grouped by peak/valley/neutral until change is obvserved
        self.btc_train_grouped = defaultdict(list)
        self.count = 0
        for i, x in enumerate(self.train_targets):
            if i + 1 != len(self.train_targets):
                if self.train_targets[i + 1] == x:
                   self.btc_train_grouped[self.count].append(self.train_s[i + 1])
                else:
                    self.count += 1
                    self.btc_train_grouped[self.count].append(self.train_s[i + 1])

        #  format test set
        self.btc_test_grouped = defaultdict(list)
        self.count = 0
        for i, x in enumerate(self.test_targets):
            if i + 1 != len(self.test_targets):
                if self.test_targets[i + 1] == x:
                    self.btc_test_grouped[self.count].append(self.test_s[i + 1])
                else:
                    self.count += 1
                    self.btc_test_grouped[self.count].append(self.test_s[i + 1])

        self.trainX = list(self.btc_train_grouped.values())
        self.testX = list(self.btc_test_grouped.values())

        self.trainX = [np.asarray(x) for x in self.trainX]
        self.testX = [np.asarray(x) for x in self.testX]

        self.group_train_targets = []
        for i, x in enumerate(self.train_targets):
            if len(self.group_train_targets) == 0:
                self.group_train_targets.append(self.train_targets[0])
            else:
                if x != self.group_train_targets[-1]:
                    self.group_train_targets.append(x)

        self.group_test_targets = []
        for i, x in enumerate(self.test_targets):
            if len(self.group_test_targets) == 0:
                self.group_test_targets.append(self.test_targets[0])
            else:
                if x != self.group_test_targets[-1]:
                    self.group_test_targets.append(x)

        # pad sequences for uneven time steps

        self.trainX = pad_sequences(self.trainX, maxlen=None, dtype='float32', padding='pre', truncating='pre', value=0.0)

        self.maxpad = len(self.trainX[0])
        self.testX = pad_sequences(self.testX, maxlen=self.maxpad, dtype='float32', padding='pre', truncating='pre', value=0.0)

        # reshape data for LSTM
        self.train_reshape = np.reshape(self.trainX, (len(self.trainX), self.maxpad, 5))
        self.test_reshape = np.reshape(self.testX, (len(self.testX), self.maxpad, 5))

        self.train_target_vec = []
        self.test_target_vec = []

        for x in self.group_train_targets:
            if x == 0:
                self.train_target_vec.append([1, 0, 0])
            elif x == 1:
                self.train_target_vec.append([0, 1, 0])
            elif x == 2:
                self.train_target_vec.append([0, 0, 1])

        for x in self.group_test_targets:
            if x == 0:
                self.test_target_vec.append([1, 0, 0])
            elif x == 1:
                self.test_target_vec.append([0, 1, 0])
            elif x == 2:
                self.test_target_vec.append([0, 0, 1])

        return self.trainX, self.train_target_vec, self.testX, self.test_target_vec, self.maxpad


    def batch_format(self, data, triggerPrice ,maxpad):
        ###
        # use for formatting during live training
        # labels data for you
        data , self.newTriggerPrice= self.label_perfect_trades(data, triggerPrice, new_training=False)
        #Could add the evaluation function here using data value above, which is the data peak/valley data 
        
        data.set_index('Timestamp_ms', inplace=True)
        self.targets = data['target'].values
        self.btc_df = data.drop('target', 1).values
        self.s = StandardScaler()
        self.btc_s = self.s.fit_transform(self.btc_df)

        # Format for time series        # Grouped by peak/valley/neutral until change is obvserved
        # The above notes are not true, the reality is that this takes values from the above vector
        # create a empty dictionary filled with lists, then based on the length of the targets value increments
        # and then overwrites existing data (see ln 845) and appends the new value to the dictionary 
        self.btc_train_grouped = defaultdict(list)
        self.count = 0
        for i, x in enumerate(self.targets):
            if i + 1 != len(self.targets):
                if self.targets[i + 1] == x:
                    self.btc_train_grouped[self.count].append(self.btc_s[i + 1])
                else:
                    self.count += 1
                    self.btc_train_grouped[self.count].append(self.btc_s[i + 1])



        self.trainX = list(self.btc_train_grouped.values())
        #print("TrainX:",self.trainX)

        self.trainX = [np.asarray(x) for x in self.trainX]

        self.group_train_targets = []
        for i, x in enumerate(self.targets):
            if len(self.group_train_targets) == 0:
                self.group_train_targets.append(self.targets[0])
            else:
                if x != self.group_train_targets[-1]:
                    self.group_train_targets.append(x)

        # pad sequences for uneven time steps
        self.trainX = pad_sequences(self.trainX, maxlen=103, dtype='float32', padding='pre', truncating='pre', value=0.0)

        # reshape data for LSTM
        self.train_reshape = np.reshape(self.trainX, (len(self.trainX), maxpad, 5))

        self.train_target_vec = []

        for x in self.group_train_targets:
            if x == 0:
                self.train_target_vec.append([1, 0, 0])
            elif x == 1:
                self.train_target_vec.append([0, 1, 0])
            elif x == 2:
                self.train_target_vec.append([0, 0, 1])

        return self.train_reshape, self.train_target_vec, self.newTriggerPrice
    
    def test_artificial_data(self, artificial_data, real_data):
        ##
        #Compares artificial data, and real data, used to increment the total amount of successes and 
        #   total calls, and train the AI. 
        #
        #Parameters - Artificial data -dataframe of model generated data including timestamp and string qualifier
        #           - real_data - dataframe of actual data pulled from binance
        #           - total_calls - total amount of data evaluations  
        #           - successes - successful evaluations
        #Returns    -(for testing)
        #           - successes either 0 or 1 for success or failure, will be used to increment external success
        #               value 
        #           - success rate - % success rate, if>80% considered success, else failure
        ##
        if (artificial_data['Qualifier'].iloc[-1] == "peak") and (real_data['target'].iloc[-1] == 2 ):
            self.successes += 1
            self.total_calls += 1
        elif (artificial_data['Qualifier'].iloc[-1] == "valley") and (real_data['target'].iloc[-1] == 1 ):
            self.successes += 1
            self.total_calls += 1 
        elif(artificial_data['Qualifier'].iloc[-1] == "neutral") and (real_data['target'].iloc[-1] == 0 ):
            self.successes += 1
            self.total_calls += 1
        else:
            self.total_calls += 1
        success_rate = ((self.successes/self.total_calls)*100)
        return success_rate
    
    def buy_sell_graph_setup(self):
        plt.ion()
        plt.figure(figsize=(10, 5))
    
    def buy_sell_graph(self, y_data, x, y, x_valley, y_valley, x_peak, y_peak, model_predict,pred_time):
        plt.clf()
    
        if len(model_predict) == 1:            
            if model_predict[0] == "valley":
                x_valley.append(pred_time)
                y_valley.append(y_data)
                #if binance.has['createMarketOrder']:
                    #binance.create_market_buy_order(symbol='BTC/USDT',amount=1)
                #binance.create_order(symbol='BTC/USDT',type='market',side='buy',amount=10,price=None,params={'test':True})
                #buy_sell_market('Test','BTC/USDT','buy', 1)
            if model_predict[0] == "peak":
                x_peak.append(pred_time)
                y_peak.append(y_data)
                #if binance.has['createMarketOrder']:
                    #binance.create_market_sell_order(symbol='BTC/USDT',amount=1)
                #binance.create_order(symbol='BTC/USDT',type='market',side='sell',amount=1,price=None,params={'test':True})
                #buy_sell_market('Test','BTC/USDT','sell', 1)
        if len(model_predict) == 2:
            if model_predict[1] == "valley":
                x_valley.append(pred_time)
                y_valley.append(y_data)
                #if binance.has['createMarketOrder']:
                    #binance.create_market_buy_order(symbol='BTC/USDT',amount=1)
                #binance.create_order(symbol='BTC/USDT',type='market',side='buy',amount=10,price=None,params={'test':True})
                #buy_sell_market('Test','BTC/USDT','buy', 1)
            if model_predict[1] == "peak":
                x_peak.append(pred_time)
                y_peak.append(y_data)
                #if binance.has['createMarketOrder']:
                    #binance.create_market_sell_order(symbol='BTC/USDT',amount=1)
                #binance.create_order(symbol='BTC/USDT',type='market',side='sell',amount=1,price=None,params={'test':True})
                #buy_sell_market('Test','BTC/USDT','sell', 1)
    
        #Neutral/All predictions
        plt.plot(x, y, 'b')
        #Valley predictions
        plt.scatter(x_valley, y_valley, c='r', marker='o', label='valley')
        #Peak predictions
        plt.scatter(x_peak, y_peak, c='g', marker='o', label='peak')
        
        plt.title('Prediction', fontsize=26)
        plt.xlabel('Time', fontsize=18)
        plt.xticks(rotation=45, horizontalalignment="right")
        plt.ylabel('Value', fontsize=18)
        plt.legend()
        plt.tight_layout()
        plt.pause(0.05)
        plt.draw()
        
    def buy_sell_market(self, account, symbol, side, amount, stop_loss_price=None):
    
        FEE_RATE = 0.0025 # TODO: improve fee determination
        status = "Pending"
        open_ts = time.time()
    
        ccxt_exchange = account.get_ccxt_exchange()
        ccxt_market = ccxt_exchange.market(symbol)
    
        amount_prec = ccxt_market["precision"]["amount"]
        price_prec = ccxt_market["precision"]["price"]
    
        amount = round(amount, amount_prec)
        stop_loss_price = round(stop_loss_price, price_prec)
    
    
        fee = round(amount * FEE_RATE, price_prec)
        base_change = 0
        quote_change = 0
    
        if side == "buy":
            response = ccxt_exchange.create_market_buy_order(symbol, amount)
            order_id = response["id"]
            base_change = amount
            quote_change = -amount - fee
        elif side == "sell":
            response = ccxt_exchange.create_market_sell_order(symbol, amount)
            order_id = response["id"]
            base_change = -amount
            quote_change = amount - fee
            open_sell_revenue = quote_change
        status = "Pending"
    
        return (base_change, quote_change)
    
class Arbitrage(Screen):

    def __init__(self,**kwargs):
        self.accounts = {'bitstamp': {'uid': 'kwsm6715',
                                 'api_key': 'eFrPYCe3KJ2w6aKb6eUYwvBqjdVnkmoU',
                                 'secret': '6xc0U55xj4pHJtQDRntzIobGfcoLDlfJ',
                                },
                    'binance': {'uid': 'noah13nelson@gmail.com',
                                'api_key': 'oox7xCH0BivEah7LOmfOiHylDEj6tueEArxf3rfj4tgDWFdKNNP9CCNuvKIXxrvR',
                                'secret': 'ouo5WhOL4ISrmbYJcwANcE9piDHG2KvRmQT0i6AJbqabAGSDsxBeHjKpv7RI1VIa',
                               },
                    'poloniex': {'uid': 'noah13nelson@gmail.com',
                                 'api_key': 'JHGCEVR5-VOLY72AW-ZTDH2DXP-S14TO6KY',
                                 'secret': '93475ba24b1a371bbbc07937cfdb5da3f284f389039afba9f90fe208634ff5a6faaedd79e9b2ff14cb937c0c7f9984ee1ce038827038c7dc6447359bead24b8b',
                                },
                    'yobit': {'uid': 'noah13nelson@gmail.com',
                              'api_key': '8FA4E982CAB11663C11936006F05097B',
                              'secret': '7ddf527b43a6e876e8dc0a35714bfe2d',
                             },
                    'livecoin': {'uid': 'noah13nelson@gmail.com',
                                 'api_key': 'YgpVDcsqSdar2YnNyzWhAzxraT2gcHFN',
                                 'secret': 't9xm2v8yrhYbfwxMBbQ49huTSDFPBxP3',
                                },
                    }
        self.exchanges = ['bitstamp', 'binance', 'livecoin', 'yobit', 'poloniex']
        self.stable_coins = ['USDT','TUSD','DAI']
        self.buy_id = 0 
        self.sell_id = 0
        self.b_exc = 0
        self.s_exc = 0
        super(Arbitrage,self).__init__(**kwargs)
    
    stop = threading.Event()

    def run(self):
        threading.Thread(target=self.second_thread).start()

    def second_thread(self):
        Clock.schedule_once(self.start_test, 0)

        time.sleep(5)

        time.sleep(2)

        self.stop_test()

        threading.Thread(target=self.run_final_script()).start()

    def start_test(self, *args):
        self.remove_widget(self.start_arbitrage_button)
        self.after_run_label.text = ('Please wait...')
        anim_bar = Factory.AnimWidget()
        self.anim_box.add_widget(anim_bar)
        anim = Animation(opacity=0.3, width=100, duration=0.6)
        anim += Animation(opacity=1, width=400, duration=0.8)
        anim.repeat = True
        anim.start(anim_bar)

    @mainthread
    def update_label_main(self, new_text):
        self.pre_run_label.text = new_text
    @mainthread
    def update_label_status(self, new_text):
        self.status.text = new_text
    @mainthread
    def update_label_markettobuy(self, new_text):
        self.markettobuy.text = new_text
    @mainthread
    def update_label_markettosell(self, new_text):
        self.markettosell.text = new_text
    @mainthread
    def update_label_biddatasorted(self, new_text):
        self.biddatasorted.text = new_text
    @mainthread
    def update_label_askdatasorted(self, new_text):
        self.askdatasorted.text = new_text
    @mainthread
    def update_label_return(self, new_text):
        self.returnOutput.text = new_text

    @mainthread
    def stop_test(self):
        self.after_run_label.text = ('Currently running Arbitrage!')

        self.remove_widget(self.anim_box)
        
    def identify_arbitrage(self, symbol, exchange_list):
        
        '''
        Finds lowest ask and highest bid for a currency in given exchanges
        :param symbol: (ex. BTC/USD)
        :param exchanges: pass EXCHANGES above or another list of exchanges
        :return: ((ask,ask_price),(bid, bid_price)
        '''
        self.ask_data = dict()
        self.bid_data = dict()

        for e in exchange_list:
            self.exc = getattr(ccxt, e)()
            self.exc.load_markets()
            self.symbols_allowed = self.exc.symbols

            if symbol in self.symbols_allowed:
                self.ask_bid_prices = self.exc.fetch_order_book(symbol)
                self.ask_data[e] = self.ask_bid_prices['asks'][0][0]
                self.bid_data[e] = self.ask_bid_prices['bids'][0][0]
            else:
                continue

        self.ask_data_sorted = sorted(self.ask_data.items(), key=itemgetter(1))
        self.bid_data_sorted = sorted(self.bid_data.items(), key=itemgetter(1), reverse=True)

        self.update_label_askdatasorted("Ask Data Sorted : " + str(self.ask_data_sorted))
        self.update_label_biddatasorted("Bid Data Sorted : " + str(self.bid_data_sorted))

        self.ask_market = self.ask_data_sorted[0][0]
        self.ask_price = self.ask_data_sorted[0][1]
        self.bid_market = self.bid_data_sorted[0][0]
        self.bid_price = self.bid_data_sorted[0][1]

        self.update_label_markettobuy("Market to buy from : " + str(self.ask_market))
        self.update_label_markettosell("Market to sell in : " + str(self.bid_market))

        return self.ask_market, self.bid_market, self.ask_price, self.bid_price
    def buy_sell(self, accounts, ask_market, bid_market, ask_price, bid_price, symbol, amount):
        '''
        Initiate a transaction between accounts and exchanges
        :param accounts: dictionary of exchanges that we own with the account information as their values
        :param ask_market: exchange where we will buy currency
        :param bid_market: exchange where we will sell currency
        :param symbol: (ex. BTC/USD)
        :return: Profit value after the exchange
        '''
        self.buy_exchange = getattr(ccxt, ask_market)
        self.sell_exchange = getattr(ccxt, bid_market)
        
        self.buy_account = accounts[ask_market]
        self.sell_account = accounts[bid_market]
        
        self.buy = self.buy_exchange(self.buy_account)
        self.sell = self.sell_exchange(self.sell_account)
        
        self.buy.load_markets()
        self.sell.load_markets() 
        '''Refund symbol is the symbol that has stable coin as the base currency.
        This symbol will be used to buy stable coin in sell market, send it to the buy market,
        and sell in the buy market to get back the quote currency. This will complete the arbitrage.'''
        
        self.refund_symbol = self.check_refund_possibility(self.stable_coins,symbol,ask_market,bid_market)
        
        '''Refund symbol = False if both the exchanges do not have the same stable coin!
        Else it will contain the pair of currency symbol, i.e., Satble_coin_symbol/original_quote_symbol.'''
        
        if self.refund_symbol == 'False':
     
            self.update_label_return("Arbitrage is not possible because the exchanges do not support the stable coins!")
        
        else:
        
            '''To store the price of the stable coin that will be used to create limit order for buying and selling the coins!'''
            
            self.refund_cur_price = self.sell.fetch_order_book(self.refund_symbol)['asks'][0][0]
        
        #if check_status(b_exc,s_exc,buy_id,sell_id,cancel_id,buy_stable_id,sell_stable_id):
        
            '''Check if the previous arbitrage has been completed!'''    
        
            if self.buy.has['createMarketOrder'] and self.sell.has['createMarketOrder'] and self.refund_symbol:
                self.buy_id = self.buy.create_limit_buy_order(symbol, amount, ask_price)['id'] #purchase the currency
                
                '''Create a buy order for the main arbitrage base currency.'''
                
                if transfer(ask_market, bid_market, symbol.split('/')[0], amount): #initiate the transfer and check if it returns true.
                
                    '''Create a sell order for the main arbitrage base currency.'''
                    
                    self.sell_id = self.sell.create_limit_sell_order(symbol, amount, bid_price)['id']
                    
                    '''Create a buy order for the stable coin.'''
                    
                    self.buy_stable_id = self.sell.create_limit_buy_order(self.refund_symbol, self.refund_cur_price)['id']
                    
                    if transfer(ask_market, bid_market, symbol.split('/')[0], amount):
                        
                        self.sell_stable_id = self.buy.create_limit_sell_order(self.refund_symbol, amount, self.refund_cur_price)['id']  
                    
                    self.b_exc = ask_market
                    self.s_exc = bid_market
                    #buy_cost = buy.fetch_my_trades(symbol)[-1]['cost']
                    #sell_cost = sell.fetch_my_trades(symbol)[-1]['cost']
                    #print('sell_cost = ' + str(sell_cost))
                    # 'buy_cost = ' + str(buy_cost))
                    print("Buy ID : " + str(self.buy_id))
                    print("Sell ID : " + str(self.sell_id))
                    print("Buy stable ID : " + str(self.buy_stable_id))
                    print("Sell stable ID : " + str(self.sell_stable_id))
                    
                    self.ids.returns.text = "Arbitrage successful!" #(sell_cost - buy_cost)
                
                else:
                    
                    '''Resell the bought coins in case the transfer does not occur!
                    Also use the cancel_id to make sure that the reselling of the coins is complete before
                    starting a new arbitrage.'''
                    
                    self.cancel_id = self.buy.create_limit_sell_order(symbol, amount, bid_price)['id']
                
                self.update_label_return("Transfer unsuccessful! The bought currency is sold back!")
            
            self.update_label_return("Arbitrage could not be done as some exchange does not support stable coins!")
        
        #return 'Previous arbitrage left to be completed!'
    def transfer(self, from_market, to_market, curr, amt):
        '''
        :param from_account: name of exchange we are transferring from
        :param to_account: name of exchange we are transferring to
        :param curr: name of the currency
        :param amt: amount
        :return: return success or error and balance
        '''
        def timing(self):
            
            '''re run this function until the transfer is complete'''
            if check_balance(accounts, to_market, curr)[1] != self.to_balance_prior_transfer:
                return True
            return False
        if from_market == 'binance':
            self.update_label_return("Cannot transfer from binance")
            return False
        self.from_exchange = getattr(ccxt, from_market)
        self.from_account = self.from_exchange(self.accounts[from_market])
        self.to_exchange = getattr(ccxt, to_market)
        self.to_account = self.to_exchange(self.accounts[to_market])
        
        self.from_account.load_markets()
        self.to_account.load_markets()
        
        self.to_balance_prior_transfer = check_balance(self.accounts, to_market, curr)
        
        self.to = self.to_account.fetch_deposit_address(curr)
        self.add = self.to['address']
        self.from_account.withdraw(curr, amt, self.add)
        
        self.start = self.time.time()
        while True:
            self.end = self.time.time()
            if timing():
                return True
            elif self.end - self.start > 3600: #1 hour
                return False

    def check_balance(self, accounts, exchange, curr):
        '''
        Check the balance of an account in an exchange
        :param account: list of accounts above
        :param exchange: exchange with the account we are checking
        :return: information and balance in a tuple structure.
        '''

        self.info = getattr(ccxt, exchange)
        self.info = self.info(accounts[exchange])
        self.info.load_markets()
        self.info = self.info.fetch_balance()

        return (curr, self.info['total']['{}'.format(curr)])
        
    def check_status(self, b_ex, s_ex, b_id, s_id, c_id, b_stable_id, s_stable_id):
        
            if b_ex and s_ex:
                self.b = getattr(ccxt, b_ex)(self.accounts[b_ex])
                self.s = getattr(ccxt,s_ex)(self.account[s_ex])
                self.b.load_markets()
                self.s.load_markets()
            else:
                return True
            
            if b_id:
                
                self.b_status = self.b.fetchOrder(b_id)['status']
            
            else:
                
                self.b_status = False
            
            if s_id:
                
                self.s_status = self.s.fetchOrder(s_id)['status']
            
            else:
                
                self.s_status = False
            
            if b_stable_id:
                
                self.b_stable_status = self.s.fetchOrder(b_stable_id)['status']
            
            else:
                
                self.b_stable_status = False
            
            if s_stable_id:
                
                self.s_stable_status = self.b.fetchOrder(s_stable_id)['status']
            
            else:
                
                self.s_stable_status = False
            
            '''If buy and sell orders for the normal coin and the stable coin are filled return True to indicate end of arbitrage.'''
            
            if self.b_status == 'closed' and self.s_status == 'closed' and self.b_stable_status == 'closed' and  self.s_stable_status == 'closed':
                
                '''After all the orders are filled(i.e., the arbitrage is complete) we set the order ids and the order status 
                to false! It is to show that we have completed the arbitrage and are ready for the next arbitrage!'''
                
                self.buy_id = False
                self.sell_id = False
                self.buy_stable_id = False
                self.sell_stable_id = False
                self.b_status = False
                self.s_status = False
                self.b_stable_status = False
                self.s_stable_status = False
                
                return True
            
            else:
                
                return False
        
            if c_id:
            
                self.c = getattr(ccxt, b_ex)(self.accounts[b_ex])
                self.c.load_markets()
                self.c_status = self.c.fetchOrder(c_id)['status']
            
                if self.c_status == 'closed':
                
                    self.cancel_id = False
                
                    return True
            
            return False
    def check_refund_possibility(self, stable_coins,symbol,b_ex,s_ex):
       
        self.base_curr = symbol.split('/')[0]
        self.quote_curr = symbol.split('/')[1]
        self.b = getattr(ccxt, b_ex)(self.accounts[b_ex])
        self.s = getattr(ccxt,s_ex)(self.accounts[s_ex])
        self.b.load_markets()
        self.s.load_markets()
        
        for a in stable_coins:
            
            '''Create a symbol where the base is stable coin and the quote is the original quote.'''
            
            self.temp = a + '/' + self.quote_curr
            
            '''Check if the symbol is supported by both the exchanges.'''
            
            if (self.temp in self.s for self.s in self.b.symbols) and (self.temp in self.b for self.b in self.s.symbols):
                
                self.refund_symbol = self.temp
            
            else:
                
                self.refund_symbol = False
        
        return self.refund_symbol
        
    def run_final_script(self):
        '''main function to run with the calls'''
        '''Define a function to initialise the order ids and exchange names to some value at the
        start of the first arbitrage to make sure that arbitrage is started for the first time without
        checking for the previous arbitrage!'''
        
        #print(check_balance(accounts,'bitstamp', 'BTC'))
        #initialize()
        self.ask_market, self.bid_market, self.ask_price, self.bid_price = self.identify_arbitrage('LTC/USD', self.exchanges)
        self.profit = self.buy_sell(self.accounts, self.ask_market, self.bid_market, self.ask_price, self.bid_price, 'LTC/USD', 0.1)
        self.update_label_status("Profit : " + str(self.profit))

class Picture(Scatter):

    source = StringProperty(None)

class Robinhood(Screen):
    value = NumericProperty(1)
    @mainthread
    def update_label_status(self, new_text):
        self.status.text = new_text
        
    def click(self):
        if self.value >= 4:
            self.update_label_status("End of images reached!")
        else:
            self.update_label_status("Slide image up, then click \"Start\" again for next image")
            picture = Picture(source="foo" + str(self.value) + ".jpg")
            self.value += 1
            self.add_widget(picture)
    
class Sentiment(Screen):

    @mainthread
    def update_label_main(self, new_text):
        self.pre_run_label.text = new_text
    @mainthread
    def update_label_status(self, new_text):
        self.status.text = new_text
    @mainthread
    def update_label_webscraper(self, new_text):
        self.webscraper.text = new_text
    @mainthread
    def stop_test(self):
        self.after_run_label.text = ('Currently running Sentiment Analysis!')

        self.remove_widget(self.anim_box)

    class sentimentAnalysis(object):
        
        def __init__(self, **kwargs):
            self.index = ['BTC','ETH','EOS','LTC','XRP','BCH',
                     'ETC','XMR','ZEC','QTUM','NEO',
                     'TRX','BTM','KEY','TRUE','XUC','XLM','DASH',
                     'USDT','BTS','BIX','OMG','AST','UBTC','WPR','BNB',
                        'PAX','ZRX','HSR','RVN','ADA','OKB','GTO','TUSD',
                        'BAT','GVT','DOGE','DENT','XIN','MITH','PHX',
                        'BCPT','GNT','SC','IOT','DLT','WTC','PAI','VET','ICX',
                        'NPXS','ONT','STORM','WAVES','AE','WAN','NCASH','LSK',
                        'XEM','ARN','HYDRO','ELF','APIS','IOST','MDA','DOCK','ETF',
                        'XVG','ZIL','QASH','QKC','KNC','TNT','POLY','MBT','HT','MGO',
                        'IOTX','SWFTC','BLZ','BTG','PAL','DNT','INT','MFT','MTH',
                        'BMX','SRN','GO','BCD','NOAH','UBEX','NAS','LINK','ZEN','MCO',
                        'DGD','BBK','ABT','YOYOW']
            self.coin_names = {'bitcoin':'BTC',
                          'ethereum':'ETH',
                          'litecoin':'LTC',
                          'bitcoin cash':'BCH',
                          'ethereum classic':'ETC',
                          'monero':'XMR',
                          'zcash':'ZEC',
                          'tron':'TRX',
                          'bytom':'BTM',
                          'selfkey':'KEY',
                          'true chain':'TRUE',
                          'exchange union':'XUC',
                          'stellar':'XLM',
                          'cardano':'ADA',
                          'Tether':'USDT',
                          'Bitshares':'BTS',
                          'BiboxCoin':'BIX',
                          'OmiseGo':'OMG',
                          'AirSwap':'AST',
                          'UnitedBitcoin':'UBTC',
                          'WePower':'WPR',
                          'Binance Coin':'BNB','Paxos Standard':'PAX','0x':'ZRX','Hshare':'HSR','Ravencoin':'RVN',
         'Cardano':'ADA','Okex':'OKB','GIFTO':'GTO','True USD':'TUSD','Basic Attention Token':'BAT','Genesis Vision':'GVT','Dogecoin':'DOGE','Innity Economics':'XIN',
         'Mithril':'MITH','Red Pulse Phoenix':'PHX','BlockMason Credit Protocol':'BCPT',
                'Golem Network Token':'GNT','Siacoin':'SC','IOTA':'IOT','Agrello Delta':'DLT',
                'Waltonchain':'WTC','Project Pai':'PAI','Vechain':'VET','ICON Project':'ICX',
                'Pundi X':'NPXS','Ontology':'ONT','Aeternity':'AE','Wanchain':'WAN','Nucleus Vision':'NCASH',
                'Lisk':'LSK','NEM':'XEM','Aeron':'ARN','Hydrogen':'HYDRO','aelf':'ELF','IOS token':'IOST',
                'Moeda':'MDA','Dock.io':'DOCK','EthereumFog':'ETF','Verge':'XVG','Zilliqa':'ZIL','Quoine Liquid':'QASH',
                'QuarkChain':'QKC','Kyber Network':'KNC','Tierion':'TNT','Polymath Network':'POLY','Multibot':'MBT',
                'Huobi Token':'HT','MobileGo':'MGO','IoTeX Network':'IOTX','SwftCoin':'SWFTC','Bluzelle':'BLZ',
                'Bitcoin Gold':'BTG','PolicyPal Network':'PAL','district0x':'DNT','Internet Node Token':'INT',
                'Mainframe':'MFT','Monetha':'MTH','BitMart Coin':'BMX','SirinLabs':'SRN','GoChain':'GO',
                'Bitcoin Diamond':'BCD','NOAHCOIN':'NOAH','Nebulas':'NAS','ChainLink':'LINK','Horizen':'ZEN',
                'Crypto.com':'MCO','Digix DAO':'DGD','BitBlocks':'BBK','ArcBlock':'ABT'}
            for i in self.index:
                self.coin_names[i.lower()] = i
            
            self.sentiment = pd.DataFrame(np.empty((len(self.index),6),dtype=object),index = self.index,
                                          columns = ['Last_News_Updated_Time','Current_sentiment','Last_sentiment',
                                                     'Current_Overall_Market_sentiment','Last_Overall_Market_sentiment','Market_Price'])
            self.sentiment.Market_price = 0
            self.sentiment.Last_News_Updated_Time = None
            self.overall_market_w = 0.3
            self.last_senti_w = 0.3
            self.last_overall_market_w = 0.3
            self.overall_market = None
            self.last_overall_market = None

        def sentence_tokenizer(self, article_list):
            sent_tokenize_list = []
            for text in article_list:
                text = text.lower()
                sent_tokenize_list.append(sent_tokenize(text))
            return sent_tokenize_list
        
        def calculate(self,data_s):
            news_articles = list(data_s.Text)
            news_sentences_list = self.sentence_tokenizer(news_articles)
            sid = SentimentIntensityAnalyzer()
            vader_sent = defaultdict(list)
            data_s_senti_list = []
            for j in range(len(news_sentences_list)):
                article = news_sentences_list[j]
                text = ' '.join(article)
                show_keys = [key for key in self.coin_names.keys() if key in text]
                tmp_coin_senti = defaultdict(list)
                
                for idx in range(len(article)):
                    current_sentence = article[idx]
                    sentiment = sid.polarity_scores(current_sentence)['compound']
                    #if sentiment is zero, we don't use it
                    if sentiment == 0:
                        continue
                    if len(show_keys) > 0:
                        for key in show_keys:
                            if key in current_sentence:
                                vader_sent[self.coin_names[key]].append(sentiment)
                                tmp_coin_senti[self.coin_names[key]].append(sentiment)
                                step = idx + 1
                                #check next 3 sentences' sentiments and add them to the current key.
                                while step < len(article) and step - idx < 4:
                                    current_sentence = article[step]
                                    sentiment = sid.polarity_scores(current_sentence)['compound']
                                    #if sentiment is zero, we don't use it
                                    if sentiment == 0:
                                        step += 1
                                        continue
                                    vader_sent[self.coin_names[key]].append(sentiment)
                                    tmp_coin_senti[self.coin_names[key]].append(sentiment)
                                    step += 1
                            else:
                                vader_sent['overall_market'].append(sentiment)
                                tmp_coin_senti['overall_market'].append(sentiment)
                    else:
                        vader_sent['overall_market'].append(sentiment)
                        tmp_coin_senti['overall_market'].append(sentiment)
                    
                mean_dict = defaultdict(float)
                for k, v in tmp_coin_senti.items():
                    mean_dict[k] = float('%.3f'%np.array(v).mean())
                data_s_senti_list.append(mean_dict)
            data_s['Sentiment'] = np.array(data_s_senti_list)                    
            output = defaultdict(float)
            #mean of each sentences
            for key, value in vader_sent.items():
                output[key] = float('%.3f'%np.array(value).mean())
            #mean of each articles
    # =============================================================================
    #         for key, value in vader_sent.items():
    #              output[key] = float('%.3f'%(np.array(value).sum()/len(news_sentences_list))) #too big!
    # =============================================================================
            
                
            #update overall market sentiment
            self.last_overall_market = self.overall_market
            if self.last_overall_market:
                self.overall_market = float('%.3f'%(output['overall_market'] * (1-self.last_overall_market_w) + self.last_overall_market * self.last_overall_market_w))
            else:
                self.overall_market = output['overall_market']
            output.pop('overall_market', None)
            return output, data_s
        
        def senti_analyzer(self,data_list):
            flat_list = [item for sublist in data_list for item in sublist]
            data = pd.DataFrame(flat_list,columns = ['Title','Text','Date','Source'])
            the_time = data.Date[0]
            #data_s =  data.sort_values(by = "Date")
            data = data.reset_index(drop=True)
            current_sentiment_dict, news_data = self.calculate(data)
            #update columns of overall market
            self.sentiment.Current_Overall_Market_sentiment = self.overall_market
            self.sentiment.Last_Overall_Market_sentiment = self.last_overall_market
            for idx in self.sentiment.index:
                # if there are new articles about this coin, update it with new sentiment
                
                last_senti = self.sentiment.loc[idx]
                
                if idx in current_sentiment_dict.keys():
                    value = current_sentiment_dict[idx]
                    if last_senti[1] == None:
                        curr_sentiment = float('%.3f'%(value * (1-self.overall_market_w) + self.overall_market_w * self.overall_market))
                        self.sentiment.loc[idx,:3] = np.array([the_time, curr_sentiment, None])
                    else:
                        #update current sentiment
                        t = float(value) * float(1- self.last_senti_w)
                        tt = float(last_senti[1]) * float(self.last_senti_w)
                        curr_sentiment_weighted = t + tt
                        curr_sentiment = float('%.3f'%(curr_sentiment_weighted * (1-self.overall_market_w) + self.overall_market_w * self.overall_market))
                        self.sentiment.loc[idx,:3] = np.array([the_time, curr_sentiment, float(last_senti[1])])
                else:
                    #If the coin doesn't show up in these recent articles then update it with overall market sentiment
                    #If it has been assigned
                    if last_senti[1]:
                        #if it was assigned with overall market sentiment, then again fill it with market sentiment
                        if float(last_senti[1]) == last_senti[4]:
                            self.sentiment.loc[idx,1:3] = np.array([self.overall_market, float(last_senti[1])])
                        #if it was different from overall market value, then we can update it with the affect of overall market sentiment
                        else:
                            t = float(last_senti[1]) * float(1-self.overall_market_w)
                            tt = float(self.overall_market_w)*float(self.overall_market)
                            curr_sentiment = float('%.3f'%(t + tt))
                            self.sentiment.loc[idx,1:3] = np.array([curr_sentiment, float(last_senti[1])])
                    #If it has never been assigned, use market sentiment to fill
                    else:
                        self.sentiment.loc[idx,1:3] = np.array([self.overall_market, last_senti[1]])
            
            return news_data
        
                    

    class news_class():
        
        def __init__(self, current_time, website, stop, k=None):
            self.news_latest = ""
            self.current_time = current_time
            self.is_exist = False
            self.website = website
            #stop 10s after first scrape
            self.freq1 = 10
            #stop 30s
            self.freq2 = 30
            self.stop = stop
            self.news_latest = None       
            if website in ["thebitcoinnews","newsbtc","cryptoslate","coinstaker","btcwires","bitcoinist","ccn"]:
                self.change_text = True
            else:
                self.change_text = False
            self.k = k
                
        def clean_text(text):
            pass
        
        def get_news(self, links):
            news_text = []
            self.news_latest = links[0]
            for link in links:
                date = self.current_time
                try:
                    article = Article(link.strip())
                    article.download()
                    article.parse()
                    text = article.text
                    if self.change_text:
                        text = self.clean_text(text)
                    if text:
                        news_text.append([article.title,text,date,self.website])
                except Exception as e: 
                    print(e)
                    pass
            return news_text
        
        def write_into_mysql():
            pass
        
        def scraper(self, stop):
            pass
            
        def process_news(self, links):
        
            news = links
            if self.stop:
                print("{} Webscraper started time:{}\nWrote {} news".format(self.website, self.current_time ,len(news)))
                return news
            else:
                self.write_into_csv(news)
                print("{} Webscraper started time:{}\nWrote {} news".format(self.website, self.current_time ,len(news)))
        
        
    class thebitcoinnews(news_class):
        def scraper(self):
            links = []
            url = 'https://thebitcoinnews.com/category/bitcoin-news/'
            try:
                r = requests.get(url)
                soup = BeautifulSoup(r.text, 'html.parser')
            
                for post in soup.find_all('div', class_ = 'td-block-span6'):
                    link = post.find('a').get('href')
                    if link == self.news_latest:
                        break
                    else:
                        links.append(link)
                if self.stop:
                    self.update_label_webscraper(self.process_news(links))
                    return self.process_news(links)
            except Exception as e: print(e)
                    
            
        
        def clean_text(self, texxt):
            content = texxt.split("\n\n")
            keep = len(content)
            if keep < 5:
                return None
            if 'source' == content[-1][:6]:
                keep -= 2
            try:
                for i in range(len(content) - 1, len(content) - 8, -1):
                    if content[i] == "For the latest cryptocurrency news, join our Telegram!":
                        keep = i
                        break
            except Exception:
                pass
            return("\n\n".join(content[:keep]))

    class newsbtc(news_class):
        def clean_text(self,texxt):
            content = texxt.split("\n\n")
            keep = len(content)
            if 'Featured' == content[-1][:8]:
                keep -= 1
            elif 'Previous' == content[-1][:8]:
                keep -= 2
            return("\n\n".join(content[:keep]))
        
        
        def scraper(self):
            links = []
            urls = ['https://www.newsbtc.com/category/bitcoin','https://www.newsbtc.com/category/crypto/',
                   'https://www.newsbtc.com/category/crypto-tech/','https://www.newsbtc.com/category/industry-news/']
            try:
                r = requests.get(urls[self.k])
                soup = BeautifulSoup(r.text, 'html.parser')
            
                posts = soup.find('div', class_ = 'row posts')
                for post in posts.find_all('div', class_ = 'post-content'):
                    link = post.find('a', class_ = 'link').get('href')
                    if link == self.news_latest:
                        break
                    else:
                        links.append(link)
                if self.stop:
                    return self.process_news(links)
            except Exception as e: print(e)

    class cryptovest(news_class):
        def scraper(self):
            url = 'https://cryptovest.com/tag/bitcoin-news/'
            try:
                r = requests.get(url)
                soup = BeautifulSoup(r.text, 'html.parser')
                links = []
            
                for post in soup.find_all('div','col-12 col-md-6 col-lg-6 p--8 post'):
                    link = 'https://cryptovest.com' + post.find('a').get('href')
                    if link == self.news_latest:
                        break
                    else:
                        links.append(link)
                if self.stop:
                    return self.process_news(links)
            except Exception as e: print(e)
            

    class cryptoslate(news_class):
        def scraper(self):
            links = []
            try:
                url = 'https://cryptoslate.com/news'
                r = requests.get(url)
                soup = BeautifulSoup(r.text, 'html.parser')
            
                for post in soup.find_all('div', class_ = 'post small-post'):
                    link = post.find('h2').find("a").get('href')
                    if link == self.news_latest:
                        break
                    else:
                        links.append(link)
                if self.stop:
                    return self.process_news(links)
            except Exception as e: print(e)
        
        def clean_text(self, texxt):
            content = texxt.split("\n\n")
            content = content[:-2]
            if content[-1][-13:] == "CryptoCompare":
                content = content[:-1]
            return("\n\n".join(content))
        

    class cointelegraph(news_class):
        def scraper(self):
               base_url = "https://cointelegraph.com"
               links = []
               try:
                   r = requests.get(base_url)
                   soup = BeautifulSoup(r.text, 'html.parser')
                   recent = soup.find('section', id = 'post-content').find('div',class_ = 'row')
                   for post in recent.find_all('div',class_='post boxed'):
                       link = post.find('a').get('href')
                       if link == self.news_latest:
                           break
                       else:
                           links.append(link)
                   if self.stop:
                       return self.process_news(links)
                   
               except Exception as e: print(e)

    class coinstaker(news_class):
        def scraper(self):
            url = 'https://www.coinstaker.com'
            try:
                r = requests.get(url)
                soup = BeautifulSoup(r.text, 'html.parser')
                links = []
            
                for post in soup.find_all('div',class_ = 'blogpost'):
                    link = post.find('a').get('href')
                    if link == self.news_latest:
                        break
                    else:
                        links.append(link)
                if self.stop:
                    return self.process_news(links)
            except Exception as e: print(e)
                        
        def clean_text(self,texxt):
            content = texxt.strip().split("\n\n")
            keep = len(content)
            idx = keep
            while idx > 0 and idx > (len(content) - 6):
                if 'Read More' == content[idx - 1][:9]:
                    keep = idx - 1
                if 'You can also' == content[idx - 1][:12]:
                    keep = idx - 1
                if 'Join us' == content[idx - 1][:7]:
                    keep = idx - 1
                idx -= 1
            return("\n\n".join(content[:keep]))
            
    class coinspeaker(news_class):
        def scraper(self):
            url = 'https://www.coinspeaker.com'
            try:
                r = requests.get(url)
                soup = BeautifulSoup(r.text, 'html.parser')
                links = []
            
                section = soup.find('div',class_ = 'sectionContent')
                for post in section.find_all('div',class_ = 'itemBlock'):
                    link = post.find('a').get('href')
                    if link == self.news_latest:
                        break
                    else:
                        links.append(link)
                if self.stop:
                    return self.process_news(links)
            except Exception as e: print(e)

    class coindesk(news_class):
        def scraper(self):
            
            url = "https://www.coindesk.com/"
            try:
                r = requests.get(url)
                soup = BeautifulSoup(r.text, 'html.parser')
                links = []
            
                for post in soup.find_all('div',class_ = "post-info"):
                    link = post.find('a').get('href')
                    if link == self.news_latest:
                        break
                    else:
                        links.append(link)
                if self.stop:
                    return self.process_news(links)
            except Exception as e: print(e)

    class ccn(news_class):
        def scraper(self):
            url = "https://www.ccn.com/"
            links=[]
            try:
                headers={'User-Agent': 'Mozilla/5.0'}
                r = requests.get(url, headers = headers)
                soup = BeautifulSoup(r.text, 'html.parser')
                posts = soup.find('div','posts-row')
                for post in posts.find_all('article'):
                    link = post.find('a').get('href')
                    if link == self.news_latest:
                        break
                    else:
                        links.append(link)
                        
                if self.stop:
                    return self.process_news(links)
            except Exception as e: print(e)
                
                    
        def clean_text(self,texxt):
            content = texxt.split("\n\n")
            content = content[1:-3]
            return("\n\n".join(content))
            
    class btcwires(news_class):
        def scraper(self):
              
           url = "https://www.btcwires.com/"   
           try:
               r = requests.get(url)
               soup = BeautifulSoup(r.text, 'html.parser')
               links = []
           
               main_posts = soup.find("div", class_ = "home-post")
               for post in main_posts.find_all("div",class_ = "post-box"):
                   try:
                       detail = post.find('div',class_ = 'post-details')
                       link = detail.find('a').get('href')
                       if link == self.news_latest:
                           break
                       else:
                           links.append(link)
                   except:
                       continue
                
               if self.stop:
                   return self.process_news(links)
           except Exception as e: print(e)
            
        def clean_text(self, texxt):
            content = texxt.strip().split("\n\n")
            keep = len(content)
            return("\n\n".join(content[1:keep]))
        

    class bitcoinist(news_class):

        def scraper(self):
            links= []
            url = 'https://bitcoinist.com/latest-news'
            try:
                r = requests.get(url)
                soup = BeautifulSoup(r.text, 'html.parser')
            
                for post in soup.find_all('div', class_ = 'news-content cf'):
                    news = post.find('h3', class_ = 'title')
                    link = news.find('a').get('href')
                    if link == self.news_latest:
                           break
                    else:
                           links.append(link)
                    
                if self.stop:
                    return self.process_news(links)
            except Exception as e: print(e)
        
        def clean_text(self, texxt):
            content = texxt.strip().split("\n\n")
            keep = len(content)
            return("\n\n".join(content[:keep-3]))


    def starter(self, object):
        
        self.current_time = datetime.datetime.now()
        #self.time_str = self.current_time.strftime("%Y_%m_%d_%H")
        scaper_btcwires = self.btcwires(self.current_time,"btcwires",True)
        scaper_bitcoinist = self.bitcoinist(self.current_time,"bitcoinist",True)
        scaper_coindesk = self.coindesk(self.current_time,"coindesk",True)
        scaper_ccn = self.ccn(self.current_time,"ccn",True)
        scaper_coindesk = self.coindesk(self.current_time,"coindesk",True)
        scaper_coinspeaker = self.coinspeaker(self.current_time,"coinspeaker",True)
        scaper_coinstaker = self.coinstaker(self.current_time,"coinstaker",True)
        scaper_cointelegraph = self.cointelegraph(self.current_time,"cointelegraph",True)
        scaper_cryptovest = self.cryptovest(self.current_time,"cryptovest",True)
        scaper_cryptoslate = self.cryptoslate(self.current_time,"cryptoslate",True)
        scaper_thebitcoinnews = self.thebitcoinnews(self.current_time,"thebitcoinnews",True)
        scaper_newsbtc_1 = self.newsbtc(self.current_time,"newsbtc",True,0)
        scaper_newsbtc_2 = self.newsbtc(self.current_time,"newsbtc",True,1)
        scaper_newsbtc_3 = self.newsbtc(self.current_time,"newsbtc",True,2)
        scaper_newsbtc_4 = self.newsbtc(self.current_time,"newsbtc",True,3)
        self.scrapers = [scaper_bitcoinist,scaper_btcwires,scaper_ccn,scaper_coindesk,
                         scaper_coinspeaker,scaper_coinstaker,scaper_cointelegraph,scaper_cryptovest,
                         scaper_cryptoslate,scaper_thebitcoinnews,scaper_newsbtc_1,scaper_newsbtc_2,
                         scaper_newsbtc_3,scaper_newsbtc_4]            
        #self.scrapers = [scaper_bitcoinist,scaper_btcwires]
        self.curr_sentiment = self.sentimentAnalysis()
        

    
    def report_start(self):
        self.current_time = datetime.datetime.now()
        print("______________________________")
        self.update_label_status("Sentiment analyzer starts on {}".format(self.current_time))
        #self.time_str = self.current_time.strftime("%Y_%m_%d_%H")      

    def scrape(self):
        total_news = []
        for cur_scraper in self.scrapers:
            cur_scraper.current_time = self.current_time
            news = cur_scraper.scraper()
            if news != None and len(news) > 0:
                total_news.append(news)
        
        if len(total_news) > 0:
            news_df = self.curr_sentiment.senti_analyzer(total_news)
            with pd.option_context('display.max_rows', None, 'display.max_columns',None,'display.width',5000):
                print(news_df.iloc[:,[0,2,4]])
                
            with pd.option_context('display.max_rows', None, 'display.max_columns',None,'display.width',5000):
                print(self.curr_sentiment.sentiment)
            return 1
        else:
            print("No valid news to update.")
            return 0
        
            
    def report_end(self):
        self.current_time = datetime.datetime.now()
        self.update_label_status("Sentiment analyzer ends on {}".format(self.current_time))


    def run_final_script(self):

        self.st = self.starter(True)

        while 1:
            self.c_time = datetime.datetime.now()
            self.st = self.report_start()
            self.st = self.scrape()
            self.isUpdated = self.st
            if self.isUpdated:
                save_object(self.st, "scraper_starter_noDB.pkl")
            #time.sleep(900)
            self.st.report_end()
            self.now = datetime.datetime.now()
            print("Time spent:{}".format(self.now - self.c_time))

    def save_object(obj, filename):
        with open(filename, 'wb') as output:  # Overwrites any existing file.
            pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)
        output.close()
        print("Scraper saved successfully!")
        
    stop = threading.Event()

    def run(self):
        threading.Thread(target=self.second_thread).start()

    def second_thread(self):
        Clock.schedule_once(self.start_test, 0)

        time.sleep(5)

        time.sleep(2)

        self.stop_test()

        threading.Thread(target=self.run_final_script()).start()

    def start_test(self, *args):
        self.remove_widget(self.start_sentiment_button)
        self.after_run_label.text = ('Please wait...')
        anim_bar = Factory.AnimWidget()
        self.anim_box.add_widget(anim_bar)
        anim = Animation(opacity=0.3, width=100, duration=0.6)
        anim += Animation(opacity=1, width=400, duration=0.8)
        anim.repeat = True
        anim.start(anim_bar)


class PullPortfolio(Screen):
     portfolio = ObjectProperty()    
     list_of_coins = []
     stop = threading.Event()
     
     
     total_value_USDT = 0
     total_value_BTC = 0
     btc_to_usd_small_coins = 0
     final_total_USDT = 0

     
     def run_portfolio_evaluation(self):
        threading.Thread(target=self.second_thread).start()
        
     def second_thread(self):
         #Schedules threading event for updating portfolio info every 2 min.
         #Avoids bans this way, and its 
         #Clock.schedule_interval(lambda dt: self.run_portfolio_evaluation(), 120)

         threading.Thread(target=self.portfolio_evaluation()).start()
    
    
     def portfolio_evaluation(self):
        print("Pulling Portfolio Data")

       
        def get_ticker_data(exchange = ccxt.binance({'ratelimit':60000, 'enableRateLimit': True}), symbol='XRP/USDT', ticker_key_value='bid,ask'): #updated fucntion to take multiple values
            list_of_symbols = symbol.split(",")
            data = []
            if (len(list_of_symbols) > 1):
                for symbol in list_of_symbols:
                    ticker_data=exchange.fetchTicker(symbol) #Fetches ticker data based on exchange, and symbol
                    if (ticker_key_value == ''): # prints entire ticker
                        return ticker_data 
                    else: #Prints only selected ticker key values
                        list_of_keys = ticker_key_value.split(",")
                        #Loop through the new list of keys
                        for key in list_of_keys:
                            if key in ticker_data.keys():
                                data.append(ticker_data.get(key)) #  this is for an array
                            else:
                                print("An incorrect key was entered, see documentation.")
                return data
            else:
                ticker_data=exchange.fetchTicker(symbol) #Fetches ticker data based on exchange, and symbol
                data = []
                if (ticker_key_value == ''): # prints entire ticker
                    return ticker_data 
                else: #Prints only selected ticker key values
                    list_of_keys = ticker_key_value.split(",")
                    #Loop through the new list of keys
                    for key in list_of_keys:
                        if key in ticker_data.keys():
                            data.append(ticker_data.get(key)) #  this is for an array
                        else:
                            print("An incorrect key was entered, see documentation.")
                return data
            
        
            #Gets Current Portfolio Value in Coins
        def get_order_details():
            #gave my details for now thats the reason why no orders or trades are printed
            exchange = ccxt.binance({'ratelimit': 60000, 'enableRateLimit': True,
                                     'api_key': 'BLQQZAD2oSZEOhxDtRW5DNhBtNMQvE4uqJJ5flGwjoqc6xMIvwANV3ICAFygTiIZ',
                                     'secret': 'YHWx0PtFiHnUSKc3XDbHwNXIb2LGPcm0XYMzWgwm5ra9kx5Xxp8lKfhVdzIKw3yQ',
                                     
                                     'options':{'adjustForTimeDifference':True},
                                     })
    #'recvWindow': 1000000000000,'nonce': ccxt.Exchange.milliseconds
            values = exchange.fetch_balance() #All coins carried by the exchange
            list_of_coins = [] # List of my coins
            
            total_value_USDT = 0 #Used to calculate the total value of a portfolio
            total_value_BTC = 0
            total_value_BTC_small_coins = 0
            
            for coin in values:
                if coin in values.keys():
                    organizational_array = [] #Used for organization into nested arrays
                    coin_values = values.get(coin) #Assign the Free, used and total values for measurement 
                    #Checks value of "total" parameter fetched, and appends to organizational_array
                    if coin_values.get('total') != None and coin_values.get('total') > .001: #Checking to see if a coin is not zero unfortunately this includes coins with "none" value
                        organizational_array.append(coin) #Append coin to the array
                        organizational_array.append(coin_values.get('total'))#append total amount of coin                       
                        #Fetch Current Pricing Information for coins in account
                        try: 
                            ticker_data = get_ticker_data(symbol=str(coin+"/USDT"))
                            #Calculate Current Value of Portfolio Based on Bid/Ask for USDT the default value
                            organizational_array.append("USDT: {0:.4f}".format(float(((ticker_data[0]+ticker_data[1])/2)*coin_values.get('total'))))
                            total_value_USDT +=(((ticker_data[0]+ticker_data[1])/2)*coin_values.get('total'))
                        except:
                            try:
                                ticker_data = get_ticker_data(symbol=str(coin+"/BTC"))
                                #Calculate Current Value of Portfolio Based on Bid/Ask for BTC due to USDT not being able to trade
                                if coin == "USDT":
                                    organizational_array.append("USDT: {0:.4f}".format(float(coin_values.get('total'))))
                                    total_value_USDT += (coin_values.get('total'))
                                else:
                                    organizational_array.append("BTC: {0:.4f}".format(float(((ticker_data[0]+ticker_data[1])/2)*coin_values.get('total'))))
                                    total_value_BTC += (((ticker_data[0]+ticker_data[1])/2)*coin_values.get('total'))
                            except: 
                                try:
                                         #This should only include BTC/USDT
                                    ticker_data = get_ticker_data(symbol=str("BTC/"+coin))
                                    if coin == "USDT": #Checks if the coin is USDT, and appends USDT value to it
                                        organizational_array.append("USDT: {0:.4f}".format(float(coin_values.get('total'))))
                                        total_value_USDT += (coin_values.get('total'))
                                    else:
                                        organizational_array.append("BTC: {0:.4f}".format(float(((ticker_data[0]+ticker_data[1])/2)*coin_values.get('total'))))
                                        total_value_BTC += (((ticker_data[0]+ticker_data[1])/2)*coin_values.get('total'))
                                except:
                                    organizational_array.append("N/A")
                                
                        organizational_array.append(ticker_data[0]) #Appends values to internal array
                        organizational_array.append(ticker_data[1]) #These will be at the end of list_of_coins

                        list_of_coins.append(organizational_array) 
                        
                    elif coin_values.get('total') != None and coin_values.get('total') < .0001 and coin_values.get('total') != 0:
                            try:
                                ticker_data = get_ticker_data(symbol=str(coin+"/USDT"))
                                #Calculate Current Value of Portfolio Based on Bid/Ask for USDT the default value
                                organizational_array.append("USDT: {0:.4f}".format(float(((ticker_data[0]+ticker_data[1])/2)*coin_values.get('total'))))
                                total_value_USDT +=(((ticker_data[0]+ticker_data[1])/2)*coin_values.get('total'))
                              
                            except: 
                                 try: 
                                      ticker_data = get_ticker_data(symbol=str(coin+"/BTC"))
                                      #Calculate Current Value of Portfolio Based on Bid/Ask for BTC due to USDT not being able to trade
                                      organizational_array.append("BTC: {0:.5f}".format(float(((ticker_data[0]+ticker_data[1])/2)*coin_values.get('total'))))
                                      total_value_BTC_small_coins += (((ticker_data[0]+ticker_data[1])/2)*coin_values.get('total'))
                                 except: 
                                    print(coin+"This coin cannot be propperly quantified.")
                            
            #calculate the USDT value of total_value_BTC_small_coins
            an_array = get_ticker_data(symbol="BTC/USDT")
            btc_to_usd_small_coins = ((sum(an_array))/2)*total_value_BTC_small_coins

            #Calculate the USDT Value of the Entire Wallet            
            tickerArray = get_ticker_data(symbol="BTC/USDT") #Get ticker data
            ticker_BTCtoUSDT =((sum(tickerArray))/2) #Converts Bid/Ask price to average
            final_total_USDT = btc_to_usd_small_coins+total_value_USDT+( total_value_BTC * ticker_BTCtoUSDT ) #Converts BTC value of coin to USDT, and adds in coins less than .001
                    
            portfolio_info_df = pd.DataFrame(list_of_coins, columns=['Symbol','Amount Owned','Current Value','Current Bid','Current Ask'])
            print (portfolio_info_df)
            
            #Assignment of Kivy Values From df
            self.display.text = str(portfolio_info_df.iloc[0])
            self.display_one.text = str(portfolio_info_df.iloc[1])
            self.display_two.text = str(portfolio_info_df.iloc[2])
            self.display_three.text = str(portfolio_info_df.iloc[3]) 
            self.display_four.text = str(portfolio_info_df.iloc[4])
            self.display_five.text = str(portfolio_info_df.iloc[5]) 
            self.display_six.text = str(portfolio_info_df.iloc[6])  
            self.display_seven.text = str(portfolio_info_df.iloc[7]) 
            self.display_eight.text = str(portfolio_info_df.iloc[8]) 
            
            self.display_total.text = "Total Value BTC:"+str("{0:.6f}".format(float(total_value_BTC)))+"\nTotal Value USDT(Coins <.001):"+str("{0:.4f}".format(float( btc_to_usd_small_coins)))+"\nFinal Total Value USDT:"+str("{0:.4f}".format(float(final_total_USDT)))
            #"\nTotal Value USDT:"+str("{0:.4f}".format(float(total_value_USDT)))+
           
            return portfolio_info_df
        return get_order_details()
#Fetch order details and current price

class ScreenTwo(PullPortfolio):
    display = ObjectProperty()
    stop = threading.Event()
       
    def run_portfolio_evaluation(self):
        threading.Thread(target=self.second_thread).start()
        
    
    def second_thread(self):
         #Schedules threading event for updating portfolio info every 2 min.
         #Avoids bans this way, and its 
         #Clock.schedule_interval(lambda dt: self.run_portfolio_evaluation(), 120)
#         This is most likely only here because of the animations between loading the actual file           
#         time.sleep(5)
#         
#         time.sleep(2)
         
         threading.Thread(target=self.update_display).start()
   

    def update_display(self):
        portfolio_info_d = PullPortfolio().portfolio_evaluation()
        print("Pulling Portfolio Data")
        self.display_nine.text = str(portfolio_info_d.iloc[9])
        self.display_ten.text = str(portfolio_info_d.iloc[10])
        self.display_eleven.text = str(portfolio_info_d.iloc[11])
        self.display_twelve.text = str(portfolio_info_d.iloc[12])
        self.display_thirteen.text = str(portfolio_info_d.iloc[13])
        self.display_fourteen.text = str(portfolio_info_d.iloc[14])
        self.display_fifteen.text = str(portfolio_info_d.iloc[15])
        self.display_sixteen.text = str(portfolio_info_d.iloc[16])
        #self.display_seventeen.text = str(portfolio_info_df.iloc[17])
        #self.display_total.text = "Total Value BTC:"+str("{0:.6f}".format(float(total_value_BTC)))+"\nTotal Value USDT:"+str("{0:.4f}".format(float(total_value_USDT)))+"\nTotal Value USDT(Coins <.001):"+str("{0:.4f}".format(float( btc_to_usd_small_coins)))+"\nFinal Total Value USDT:"+str("{0:.4f}".format(float(final_total_USDT)))
    pass




class ScreenManagement(ScreenManager):
    pass

class MainScreen(Screen):
    pass


presentation = Builder.load_file("PortfolioApp.kv")
class AICryptoV2(App):

    def build(self):
        return presentation


AICryptoV2().run()