#artificial intelligence
#artificial_i


#Google colaboratoryでLinuxコマンドを実行する場合、コマンドの先頭に「！」を付ける （!pipや!wgetなど）
#事前処理
#!pip list
#!pip install -q xlrd
#!pip install pandas_datareader
#!pip install --upgrade yfinance

###############################################
# Jupyter_notebook's Shortcut
# Ctrl + \ :すべてのランタイムをリセット[←ショートカットを任意に割り振り]
# Ctrl + Enter :セルを実行
###############################################

#from google.colab import files
#from google.colab import drive
'''
import google.colab
import googleapiclient.discovery00
import googleapiclient.http
'''
import datetime
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier  # 決定木（分類）

#drive.mount('/content/gdrive')

#import pandas
from pandas_datareader import data as pdr
#import yfinance as yfin
#yfin.pdr_override()

start = datetime.date(2021, 1, 1)
end = datetime.date.today()
code = '6758'  # SONY
stock = []

#https://finance.yahoo.com/quote/6758.T/history?period1=1609473600&period2=1646798399&interval=1d&frequency=1d&filter=history
#df = pdr.get_data_yahoo('1514.T',start ='2021-06-07', end='2021-07-07')
#df = web.DataReader(f'{code}.T', 'yahoo', start, end)
adjclosed = pdr.get_data_yahoo(f'{code}.T',start, end)["Adj Close"] 
#closed = pdr.get_data_yahoo(f'{code}.T',  start, end)["Close"]  # 株価データの取得
all_data = pdr.get_data_yahoo(f'{code}.T',  start, end)  # 株価データの取得
adjclosed.to_csv('data/stocks_price_data/kabu_pre10_data.csv')  # csv書き出し
print(all_data)



'''学習'''
'''教師データの数値の配列(train_X) と結果の配列 (train_y) を学習させ、テストデータの数値の配列 (test_X) を与えると予測結果 (test_y) が帰ってくるというそれだけです。'''
'''###教師データをつくる'''
# 過去の株価と上がり下がり(結果)を学習する
# まずは一番面倒な株価の調整後終値(Adj Clouse)から教師データを作るまでのコードを用意します。
# これは終値のリストを渡すと train_X と train_y が返るようにすれば良いでしょう。


def train_data(adjclosed):  # arr = test_X
    train_X = []  # 教師データ
    train_y = []  # 上げ下げの結果の配列

    # 30 日間のデータを学習、 1 日ずつ後ろ(today方向)にずらしていく
    for i in np.arange(-30, -15):
        s = i + 14  # 14 日間の変化を素性にする
        feature = adjclosed.iloc[i:s]  # i~s行目を取り出す
        if feature[-1] < adjclosed[s]:  # その翌日、株価は上がったか？
            train_y.append(1)  # YES なら 1 を
        else:
            train_y.append(0)  # NO なら 0 を
        train_X.append(feature.values)

    # 教師データ(train_X)と上げ下げの結果(train_y)のセットを返す
    return np.array(train_X), np.array(train_y)


#%%
# 
# 
# main()
#これで train_X (教師データの配列＝学習データ) と train_y (それに対する 1 か 0 かのラベル＝結果) が返ってきます。
learning = train_data(adjclosed)  # adjclosed = test_X



'''###決定木のインスタンスを生成'''
clf = DecisionTreeClassifier(max_depth=2, random_state=0)

'''###学習させる'''
# train_X(教師データの配列＝学習データ) と train_y(それに対する 1 か 0 かのラベル＝結果) 
clf.fit(learning[0], learning[1])
#clf.fit(train_X, train_y)
#これであとは clf.predict() 関数にテストデータを渡すことで予測結果が返ってくるようになります。

'''実際に予想する'''

# 過去 30 日間のデータでテストデータを作成する
#for i in np.arange(-30, -15):
i=-15
s = i + 14

test_X = adjclosed.iloc[i:s].values  # '''テストデータの数値の配列 (test_X)'''i~s
X = np.array(test_X).reshape(-1, 14)

print("test_X= ",test_X)

#print("test_X:テストデータの数値=", X)

#clf.predict() 関数にテストデータXを渡すことで予測結果が返ってくる
results = clf.predict(X) # 予測結果 (test_y)
print("test_y:予測結果=",  clf.predict(X))

if  clf.predict(X) < 1:  # その翌日、株価は上がったか？
    res = "Decline=下落"
else:
    res = "Soaring=高騰"

print("予測結果：",res)
