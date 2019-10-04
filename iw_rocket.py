import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("housing.csv")

#入力変数と教師データに分割
#df.iloc[行,列]
x = df.iloc[:, :-1].values.astype("f")
t = df.iloc[:,-1].values.astype("f")

x.shape
t = t.reshape(len(t),1)

#データセットの準備
dataset = list(zip(x,t))

#訓練データと検証データの分割
import chainer
import chainer.functions as F
import chainer.links as L

n_train = int(len(dataset)*0.7)
n_train

train, test = chainer.datasets.split_dataset_random(dataset, n_train, seed=0)
len(train)

#モデルの定義
class NN(chainer.Chain):
    #モデルの構造を明示
    def __init__(self):
        super().__init__()
        with self.init_scope():
            self.fc1 = L.Linear(None, 1200)
            self.fc2 = L.Linear(1200, 800)
            self.fc3 = L.Linear(800, 300)
            self.fc4 = L.Linear(300, 1)
    #順伝播
    def __call__(self, x ):
        h = self.fc1(x)
        h = F.relu(h)
        h = self.fc2(h)
        h = F.relu(h)
        h = self.fc3(h)
        h = F.relu(h)
        h = self.fc4(h)
        return h

np.random.seed(0)
#インスタンス化
nn = NN()

model = L.Classifier(nn, lossfun=F.mean_squared_error)
model.compute_accuracy = False

# 学習に必要な準備
optimizer = chainer.optimizers.Adam() #確率的勾配降下法
optimizer.setup(model)
batchsize = 100

train_iter = chainer.iterators.SerialIterator(train, batchsize)
test_iter = chainer.iterators.SerialIterator(test, batchsize, repeat=False, shuffle=False)

from chainer import training
updator = training.StandardUpdater(train_iter, optimizer, device=-1)

from chainer.training import extensions
#エポックの数
epoch = 3000 #学習回数

#trainerの宣言
trainer = training.Trainer(updator, (epoch, "epoch"), out="result/housing")

#検証データで評価
trainer.extend(extensions.Evaluator(test_iter, model, device=-1))

trainer.extend(extensions.LogReport(trigger=(1, "epoch")))

#1エポックごと(trigger)にtrainデータに対するloss.accuracyとtestデータに対するloss.accuracy,経過時間を出力します
trainer.extend(extensions.PrintReport(["epoch", "main/loss", "validation/main/loss", "elapsed_time"]), trigger=(1, "epoch"))


# # 学習の実行
trainer.run()

import json
with open("result/housing/log") as f:
    logs = json.load(f)
    results = pd.DataFrame(logs)

results.head()
results[["main/loss", "validation/main/loss"]].plot()

loss = results["validation/main/loss"].values[-1]
loss

import math
math.sqrt(loss)
df.head()

from chainer import serializers,Chain
chainer.serializers.save_npz("models/housing.npz", model)

# # 学習済みモデルのロード
#モデルの構造を明示
model = L.Classifier(NN())
#モデルの読み込み
chainer.serializers.load_npz("models/housing.npz", model)

# # 予測値の計算
#今回は一番最初のサンプルに対する予測値の計算を行う
x_new = x[5]
x_new.shape
x_new = x_new[np.newaxis]
x_new.shape
y = model.predictor(x_new)

y





