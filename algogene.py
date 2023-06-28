from AlgoAPI import AlgoAPIUtil, AlgoAPI_Backtest
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
import urllib.request

instruments = ['00001HK', '00002HK', '00003HK', '00004HK', '00005HK', '00006HK',
       '00011HK', '00012HK', '00014HK', '00016HK', '00017HK', '00019HK',
       '00023HK', '00027HK', '00066HK', '00069HK', '00083HK', '00087HK',
       '00101HK', '00119HK', '00123HK', '00135HK', '00136HK', '00144HK',
       '00148HK', '00152HK', '00168HK', '00177HK', '00200HK', '00257HK',
       '00267HK', '00270HK', '00291HK', '00293HK', '00303HK', '00317HK',
       '00322HK', '00338HK', '00341HK', '00345HK', '00347HK', '00357HK',
       '00358HK', '00371HK', '00384HK', '00386HK', '00388HK', '00511HK',
       '00522HK', '00525HK', '00548HK', '00570HK', '00576HK', '00590HK',
       '00598HK', '00683HK', '00688HK', '00694HK', '00696HK', '00697HK',
       '00710HK', '00719HK', '00728HK', '00751HK', '00762HK', '00857HK',
       '00874HK', '00902HK', '00914HK', '00941HK', '00966HK', '00991HK',
       '00992HK', '00995HK', '01038HK', '01044HK', '01055HK', '01071HK',
       '01109HK', '01137HK']

threshold = 0.1
volumeCoefficient = 0.05
    
class AlgoEvent:
    def __init__(self):
        self.modelDict = {}
        
    def start(self, mEvt):
        mEvt["StartDate"] = "2010-01-01"
        mEvt["EndDate"] = "2014-01-01"
        mEvt['subscribeList'] = instruments
        mEvt['BaseCurrency'] = "HKD"
        volumeCoefficient = mEvt['InitialCapital'] / 500_000
        self.evt = AlgoAPI_Backtest.AlgoEvtHandler(self, mEvt)
        for i, instrument in enumerate(instruments):
            url = f'https://github.com/P4GAN/algogene/raw/main/pickledModels/{instrument}.pkl'
            response = urllib.request.urlopen(url)
            weights = pickle.load(response)
        
            model = tf.keras.Sequential([
                tf.keras.layers.LSTM(32, return_sequences=True, input_shape=[None, 1]),
                tf.keras.layers.Dense(1)
            ])
            
            model.set_weights(weights)
            
            self.modelDict[instrument] = model
            self.evt.consoleLog(f"Loaded model {instrument} number {i}")

        self.evt.start()
        
    def on_bulkdatafeed(self, isSync, bd, ab):
        pass

    def on_marketdatafeed(self, md, ab):
        lastprice = md.lastPrice
        # retrieve recent observations
        res = self.evt.getHistoricalBar({"instrument": md.instrument}, 50, 'D')
        closingPrices = np.array([res[t]['c'] for t in res])

        if (np.isnan(closingPrices).any()):
            return

        model = self.modelDict[md.instrument]
        
        modelPrediction = (model.predict(closingPrices[np.newaxis, :])[0, -1][0] - lastprice) / lastprice
        
        self.evt.consoleLog("model prediction = ", modelPrediction)

        if modelPrediction > threshold:
            self.sendOpenOrder(md.instrument, 1, abs(modelPrediction * volumeCoefficient))
        if modelPrediction < -threshold:
            self.sendOpenOrder(md.instrument, -1, abs(modelPrediction * volumeCoefficient))


    def on_orderfeed(self, of):
        pass

    def on_dailyPLfeed(self, pl):
        pass

    def on_openPositionfeed(self, op, oo, uo):
        pass

    def sendOpenOrder(self, instrument, buysell, vol):
        order = AlgoAPIUtil.OrderObject()
        order.instrument = instrument
        order.orderRef = 1
        order.volume = vol
        order.openclose = "open"
        order.buysell = buysell
        order.ordertype = 0 
        self.evt.sendOrder(order)