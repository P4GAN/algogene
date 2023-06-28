
from AlgoAPI import AlgoAPIUtil, AlgoAPI_Backtest
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
import urllib.request

feature_names = [
   'Norm_Returns_Daily', 'Norm_Returns_Monthly', 'Norm_Returns_Quarterly',
   'Norm_Returns_Semiannually', 'Norm_Returns_Annually', 'MACD_8_24',
   'MACD_16_48', 'MACD_32_96','Sigma_Norm'
]

instruments = ["00001HK","00002HK","00003HK","00004HK","00005HK","00006HK","00008HK","00011HK","00012HK","00013HK","00014HK","00016HK","00017HK","00019HK","00020HK","00023HK","00027HK","00066HK","00069HK","00081HK","00083HK","00087HK","00095HK","00101HK","00119HK","00123HK","00135HK","00136HK","00144HK","00148HK","00151HK","00152HK","00168HK","00175HK","00177HK","00189HK","00200HK","00220HK","00241HK","00257HK","00267HK","00268HK","00270HK","00285HK","00288HK","00291HK","00293HK","00297HK","00302HK","00303HK","00316HK","00317HK","00322HK","00338HK","00341HK","00345HK","00347HK","00354HK","00357HK","00358HK","00371HK","00384HK","00386HK","00388HK","00390HK","00392HK","00412HK","00425HK","00460HK","00467HK","00489HK","00493HK","00511HK","00512HK","00520HK","00522HK","00525HK","00546HK","00548HK","00552HK","00558HK","00570HK","00576HK","00579HK","00586HK","00590HK","00596HK","00598HK","00631HK","00639HK","00656HK","00659HK","00667HK","00669HK","00670HK","00683HK","00688HK","00694HK","00696HK","00697HK","00700HK","00710HK","00719HK","00728HK","00751HK","00753HK","00762HK","00763HK","00772HK","00777HK","00780HK","00788HK","00799HK","00811HK","00817HK","00819HK","00826HK","00836HK","00839HK","00853HK","00857HK","00867HK","00868HK","00873HK","00874HK","00880HK","00881HK","00883HK","00884HK","00902HK","00909HK","00914HK","00916HK","00921HK","00939HK","00941HK","00953HK","00956HK","00960HK","00966HK","00968HK","00973HK","00981HK","00990HK","00991HK","00992HK","00995HK","00998HK","01024HK","01030HK","01038HK","01044HK","01055HK","01060HK","01066HK","01071HK","01072HK","01088HK","01093HK","01099HK","01109HK","01112HK","01113HK","01114HK","01119HK","01128HK","01137HK","01138HK","01157HK","01167HK","01171HK","01177HK","01179HK","01186HK","01193HK","01199HK","01208HK","01209HK","01211HK","01238HK","01244HK","01258HK","01268HK","01288HK","01299HK","01302HK","01308HK","01310HK","01313HK","01316HK","01336HK","01339HK","01347HK","01357HK","01359HK","01368HK","01378HK","01385HK","01398HK","01415HK","01448HK","01458HK","01468HK","01513HK","01515HK","01516HK","01530HK","01548HK","01558HK","01563HK","01579HK","01585HK","01610HK","01618HK","01658HK","01666HK","01675HK","01691HK","01696HK","01725HK","01729HK","01735HK","01765HK","01766HK","01769HK","01772HK","01776HK","01787HK","01789HK","01797HK","01798HK","01799HK","01800HK","01801HK","01810HK","01811HK","01813HK","01816HK","01818HK","01821HK","01833HK","01839HK","01842HK","01858HK","01873HK","01876HK","01877HK","01880HK","01882HK","01883HK","01888HK","01896HK","01898HK","01907HK","01908HK","01910HK","01913HK"]


num_features = len(feature_names)

lookback_length = 30
ewmastd_span = 60

def calc_normalized_period_returns(daily_returns, daily_std, period):
    period = int(period)
    returns = (1 + daily_returns).rolling(period).apply(np.prod, raw=True) - 1
    return returns / (np.sqrt(period) * daily_std)


def calc_macd_features(price, short_period, long_period):

    short_ma = price.ewm(span=short_period, min_periods=short_period).mean()
    long_ma = price.ewm(span=long_period, min_periods=long_period).mean()
    ewmstd_63 = price.ewm(span=63).std()
    macd = short_ma - long_ma
    q = macd / ewmstd_63
    z = q / q.ewm(span=252, min_periods=252).std()

    return z

def prepareDataframe(closingPrices):
    df = pd.DataFrame({'c': closingPrices})
    
    df['Returns_Daily'] = df['c'].pct_change()
    df['Sigma'] = df['Returns_Daily'].ewm(span=ewmastd_span, min_periods=ewmastd_span).std()
    df['Norm_Returns_Daily'] = df['Returns_Daily'] / df['Sigma']
    df['Norm_Returns_Monthly'] = calc_normalized_period_returns(df['Returns_Daily'], df['Sigma'], 252 / 12)
    df['Norm_Returns_Quarterly'] = calc_normalized_period_returns(df['Returns_Daily'], df['Sigma'], 252 / 3)
    df['Norm_Returns_Semiannually'] = calc_normalized_period_returns(df['Returns_Daily'], df['Sigma'], 252 / 2)
    df['Norm_Returns_Annually'] = calc_normalized_period_returns(df['Returns_Daily'], df['Sigma'], 252)
    df['MACD_8_24'] = calc_macd_features(df['c'], 8, 24)
    df['MACD_16_48'] = calc_macd_features(df['c'], 16, 48)
    df['MACD_32_96'] = calc_macd_features(df['c'], 32, 96)
    df['Sigma_Norm'] = np.log(df['Sigma'] / df['Sigma'].rolling(181).mean())

    return df.dropna()
    
class AlgoEvent:
    def __init__(self):
        self.modelDict = {}
        
    def start(self, mEvt):
        mEvt["StartDate"] = "2010-01-01"
        mEvt["EndDate"] = "2014-01-01"
        mEvt['subscribeList'] = instruments
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

        model = self.modelDict[md.instrument]
        
        modelPrediction = model.predict(closingPrices[np.newaxis, :])[0, -1][0]
        
        if modelPrediction > 0.2:
            self.test_sendOrder(lastprice, 1, 'open')
        if modelPrediction < -0.2:
            self.test_sendOrder(lastprice, -1, 'open')

        self.evt.consoleLog("model prediction =", modelPrediction)

    def on_orderfeed(self, of):
        pass

    def on_dailyPLfeed(self, pl):
        pass

    def on_openPositionfeed(self, op, oo, uo):
        pass

    def test_sendOrder(self, lastprice, buysell, openclose):
        order = AlgoAPIUtil.OrderObject()
        order.instrument = "00001HK"
        order.orderRef = 1
        if buysell==1:
            order.takeProfitLevel = lastprice*1.1
            order.stopLossLevel = lastprice*0.9
        elif buysell==-1:
            order.takeProfitLevel = lastprice*0.9
            order.stopLossLevel = lastprice*1.1
        order.volume = 0.05
        order.openclose = openclose
        order.buysell = buysell
        order.ordertype = 0 #0=market_order, 1=limit_order, 2=stop_order
        self.evt.sendOrder(order)


