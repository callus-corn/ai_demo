import numpy as np

# -----------------------------
# 活性化関数
# -----------------------------
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_derivative(y):
    # y: sigmoid(x)
    return y * (1.0 - y)

def softmax(x):
    x = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# -----------------------------
# ネットワーク定義
# -----------------------------
class NeuralNetwork:
    def __init__(self):
        # 7 -> 16
        self.W1 = np.random.randn(7, 16) * 0.1
        self.b1 = np.zeros((1, 16))

        # 16 -> 10
        self.W2 = np.random.randn(16, 10) * 0.1
        self.b2 = np.zeros((1, 10))

    def forward(self, X):
        # 隠れ層
        self.z1 = X @ self.W1 + self.b1
        self.a1 = sigmoid(self.z1)

        # 出力層
        self.z2 = self.a1 @ self.W2 + self.b2
        self.y = softmax(self.z2)

        return self.y

    def loss(self, y_true):
        n = y_true.shape[0]
        return -np.sum(y_true * np.log(self.y + 1e-10)) / n

    def backward(self, X, y_true, lr=0.1):
        n = X.shape[0]

        # 出力層
        dz2 = (self.y - y_true) / n

        dW2 = self.a1.T @ dz2
        db2 = np.sum(dz2, axis=0, keepdims=True)

        # 隠れ層
        da1 = dz2 @ self.W2.T
        dz1 = da1 * sigmoid_derivative(self.a1)

        dW1 = X.T @ dz1
        db1 = np.sum(dz1, axis=0, keepdims=True)

        # パラメータ更新
        self.W2 -= lr * dW2
        self.b2 -= lr * db2

        self.W1 -= lr * dW1
        self.b1 -= lr * db1

# -----------------------------
# 学習データ作成
# -----------------------------
np.random.seed(0)

# 10サンプル、7特徴量
X = np.array([
    [1,1,1,1,1,1,0], # 0
    [0,1,1,0,0,0,0], # 1
    [1,1,0,1,1,0,1], # 2
    [1,1,1,1,0,0,1], # 3
    [0,1,1,0,0,1,1], # 4
    [1,0,1,1,0,1,1], # 5
    [1,0,1,1,1,1,1], # 6
    [1,1,1,0,0,0,0], # 7
    [1,1,1,1,1,1,1], # 8
    [1,1,1,1,0,1,1]  # 9
])

# クラスラベル(0～9)
labels = np.array([0,1,2,3,4,5,6,7,8,9])

# One-Hot化
Y = np.eye(10)[labels]

# -----------------------------
# 学習
# -----------------------------
net = NeuralNetwork()
epochs = 1000
for epoch in range(epochs):
    y_pred = net.forward(X)
    loss = net.loss(Y)
    net.backward(X, Y, lr=0.1)

    if epoch % 100 == 0:
        pred = np.argmax(y_pred, axis=1)
        acc = np.mean(pred == labels)

        print(
            f"Epoch {epoch:4d} "
            f"Loss={loss:.4f} "
            f"Accuracy={acc:.4f}"
        )

# -----------------------------
# 推論
# -----------------------------
np.set_printoptions(linewidth=1000)
for i in range(10):
    sample = np.array([X[i]])    
    prob = net.forward(sample)

    print("入力:", sample)
    print("予測確率:", prob)
    print("予測クラス:", np.argmax(prob))
