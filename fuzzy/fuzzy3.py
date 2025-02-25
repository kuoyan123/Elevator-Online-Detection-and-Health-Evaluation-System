import numpy as np
from scipy.signal import lfilter
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


class MembershipFunction:
    """高斯隶属函数类"""

    def __init__(self, mean, sigma):
        self.mean = mean
        self.sigma = sigma

    def compute(self, x):
        return np.exp(-((x - self.mean) ** 2) / (2 * self.sigma ** 2 + 1e-8))

    def gradient(self, x):
        """计算对参数的梯度"""
        diff = x - self.mean
        d_mean = (diff / (self.sigma ** 2 + 1e-8)) * self.compute(x)
        d_sigma = (diff ** 2 / (self.sigma ** 3 + 1e-8)) * self.compute(x)
        return d_mean, d_sigma


class MamdaniConsequent:
    """Mamdani后件模糊集"""

    def __init__(self, mean, sigma):
        self.mean = mean  # 模糊集中心
        self.sigma = sigma  # 模糊集宽度

    def compute_centroid(self, firing_strength):
        """计算去模糊化后的质心（返回与样本数相同的数组）"""
        return np.full_like(firing_strength, self.mean)  # 修正：返回数组而非标量


class TSKConsequent:
    """TSK后件线性函数"""

    def __init__(self, n_features):
        self.coef = np.random.randn(n_features + 1) * 0.1  # 包含偏置项

    def compute(self, x):
        """返回形状为(n_samples,)的输出"""
        return np.dot(x, self.coef[:-1]) + self.coef[-1]


class FuzzyRule:
    """模糊规则基类"""

    def __init__(self, antecedent_indices):
        self.antecedent_indices = antecedent_indices


class MamdaniRule(FuzzyRule):
    """Mamdani型规则"""

    def __init__(self, antecedent_indices, consequent):
        super().__init__(antecedent_indices)
        self.consequent = consequent  # MamdaniConsequent实例


class TSGRule(FuzzyRule):
    """Takagi-Sugeno型规则"""

    def __init__(self, antecedent_indices, n_features):
        super().__init__(antecedent_indices)
        self.consequent = TSKConsequent(n_features)


class HybridFIS:
    def __init__(self, n_features, expert_rules, auto_rule_count=3, lr=0.01):
        self.n_features = n_features
        self.lr = lr

        # 初始化隶属函数 (每个特征3个：低、中、高)
        self.mfs = []
        for _ in range(n_features):
            self.mfs.append([
                MembershipFunction(mean=-2.0, sigma=1.0),
                MembershipFunction(mean=0.0, sigma=1.0),
                MembershipFunction(mean=2.0, sigma=1.0)
            ])

        # 构建规则库
        self.rules = []
        # 添加专家规则(Mamdani)
        for indices, output_center in expert_rules:
            consequent = MamdaniConsequent(mean=output_center, sigma=1.0)
            self.rules.append(MamdaniRule(indices, consequent))

        # 添加自动生成的TSK规则
        for _ in range(auto_rule_count):
            indices = [np.random.randint(3) for _ in range(n_features)]
            self.rules.append(TSGRule(indices, n_features))

        # 分离规则类型
        self.mamdani_rules = [r for r in self.rules if isinstance(r, MamdaniRule)]
        self.tsk_rules = [r for r in self.rules if isinstance(r, TSGRule)]

    def compute_firing(self, X):
        """计算所有规则的触发强度"""
        n_samples = X.shape[0]
        firing_strengths = np.ones((n_samples, len(self.rules)))

        for i, rule in enumerate(self.rules):
            for feat in range(self.n_features):
                mf_idx = rule.antecedent_indices[feat]
                mf = self.mfs[feat][mf_idx]
                firing_strengths[:, i] *= mf.compute(X[:, feat])

        return firing_strengths

    def predict(self, X):
        """混合推理过程"""
        X = np.array(X)
        firing = self.compute_firing(X)
        n_samples = X.shape[0]

        # 存储各规则输出
        outputs = []
        for i, rule in enumerate(self.rules):
            if isinstance(rule, MamdaniRule):
                # Mamdani规则：返回数组形状(n_samples,)
                outputs.append(rule.consequent.compute_centroid(firing[:, i]))
            else:
                # TSK规则：返回形状(n_samples,)
                outputs.append(rule.consequent.compute(X))

        # 转换为二维数组 (n_rules, n_samples)
        outputs = np.array(outputs)  # 现在所有元素形状一致

        # 加权平均
        total_firing = np.sum(firing, axis=1, keepdims=True) + 1e-8
        weighted_output = np.sum(firing * outputs.T, axis=1) / total_firing.squeeze()
        return weighted_output

    def compute_gradients(self, X, y_true):
        """计算所有可调参数的梯度"""
        y_pred = self.predict(X)
        error = y_pred - y_true

        gradients = {
            'mfs': [{'means': [0.0] * 3, 'sigmas': [0.0] * 3} for _ in range(self.n_features)],
            'tsk_coef': [np.zeros_like(r.consequent.coef) for r in self.tsk_rules]
        }

        for n in range(X.shape[0]):
            firing = self.compute_firing(X[n:n + 1]).flatten()
            total_firing = np.sum(firing)

            for i, rule in enumerate(self.rules):
                common_term = error[n] * (firing[i] / total_firing)

                # 前件参数梯度
                for feat in range(self.n_features):
                    mf_idx = rule.antecedent_indices[feat]
                    mf = self.mfs[feat][mf_idx]
                    x_val = X[n, feat]

                    d_mean, d_sigma = mf.gradient(x_val)
                    grad_mean = common_term * d_mean
                    grad_sigma = common_term * d_sigma

                    gradients['mfs'][feat]['means'][mf_idx] += grad_mean / X.shape[0]
                    gradients['mfs'][feat]['sigmas'][mf_idx] += grad_sigma / X.shape[0]

                # TSK后件梯度
                if isinstance(rule, TSGRule):
                    x_ext = np.append(X[n], 1.0)
                    grad_coef = common_term * x_ext
                    gradients['tsk_coef'][i - len(self.mamdani_rules)] += grad_coef / X.shape[0]

        return gradients

    def update_parameters(self, gradients):
        """参数更新"""
        # 更新隶属函数参数
        for feat in range(self.n_features):
            for mf_idx in range(3):
                self.mfs[feat][mf_idx].mean -= self.lr * gradients['mfs'][feat]['means'][mf_idx]
                self.mfs[feat][mf_idx].sigma -= self.lr * gradients['mfs'][feat]['sigmas'][mf_idx]

        # 更新TSK后件参数
        for i, tsk_rule in enumerate(self.tsk_rules):
            tsk_rule.consequent.coef -= self.lr * gradients['tsk_coef'][i]

    def train(self, X, y, epochs=100, batch_size=32):
        """训练过程"""
        for epoch in range(epochs):
            indices = np.random.permutation(X.shape[0])
            for i in range(0, X.shape[0], batch_size):
                batch_idx = indices[i:i + batch_size]
                X_batch, y_batch = X[batch_idx], y[batch_idx]

                gradients = self.compute_gradients(X_batch, y_batch)
                self.update_parameters(gradients)

            if epoch % 10 == 0:
                y_pred = self.predict(X)
                mse = mean_squared_error(y, y_pred)
                print(f"Epoch {epoch}: MSE = {mse:.4f}")


# 测试验证
if __name__ == "__main__":
    expert_rules = [
        ([0, 0], -1.0),  # 特征1低 AND 特征2低 → 输出-1
        ([0, 1], -1.0),  # 特征1低 AND 特征2低 → 输出-1
        ([1, 0], -1.0),  # 特征1低 AND 特征2低 → 输出-1
        ([2, 2], 1.0)  # 特征1高 AND 特征2高 → 输出+1
    ]

    model = HybridFIS(n_features=2, expert_rules=expert_rules, auto_rule_count=3, lr=0.01)

    X, y = make_regression(n_samples=500, n_features=2, noise=0.2, random_state=42)
    y = (y - y.mean()) / y.std()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model.train(X_train, y_train, epochs=100, batch_size=32)

    y_pred = model.predict(X_test)
    print(f"\n测试集MSE: {mean_squared_error(y_test, y_pred):.4f}")

    print("\n专家规则后件中心:")
    for rule in model.mamdani_rules:
        print(rule.consequent.mean)

    print("\n自动规则TSK后件参数示例:")
    print(model.tsk_rules[0].consequent.coef)
