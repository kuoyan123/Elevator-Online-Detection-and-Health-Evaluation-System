import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


class TSKFuzzySystem:
    def __init__(self, n_rules=5, learning_rate=0.01,
                 antecedent_params=None, consequent_params=None):
        """
        初始化TSK模糊系统
        :param n_rules: 模糊规则数量
        :param learning_rate: 学习率
        :param antecedent_params: 专家提供的先验前件参数[可选]
        :param consequent_params: 专家提供的先验后件参数[可选]
        """
        self.n_rules = n_rules
        self.lr = learning_rate

        # 初始化前件参数（支持专家规则）
        self.antecedent_params = antecedent_params

        # 初始化后件参数（支持专家规则）
        self.consequent_params = consequent_params

        # 跟踪特征数量
        self.n_features = None

    def _check_initialization(self, X):
        """检查并完成系统初始化"""
        if self.antecedent_params is None:
            self._init_membership_functions(X)
        if self.consequent_params is None:
            self._init_consequent_params(X)
        if self.n_features is None:
            self.n_features = X.shape[1]

    def _init_membership_functions(self, X):
        """自动初始化前件参数（如果没有专家规则）"""
        n_features = X.shape[1]
        self.antecedent_params = []
        for i in range(n_features):
            min_val = np.min(X[:, i])
            max_val = np.max(X[:, i])
            means = np.linspace(min_val, max_val, self.n_rules)
            sigmas = np.ones(self.n_rules) * (max_val - min_val) / (2 * self.n_rules)
            self.antecedent_params.append({'means': means, 'sigmas': sigmas})

    def _init_consequent_params(self, X):
        """自动初始化后件参数"""
        n_features = X.shape[1]
        self.consequent_params = np.random.randn(self.n_rules, n_features + 1) * 0.1

    def _gaussian_mf(self, x, mean, sigma):
        """高斯隶属函数"""
        return np.exp(-((x - mean) ** 2) / (2 * sigma ** 2 + 1e-8))

    def _compute_firing_strength(self, X):
        """计算规则激活强度"""
        n_samples, n_features = X.shape
        firing_strengths = np.ones((n_samples, self.n_rules))

        for i in range(n_features):
            mf_values = np.zeros((n_samples, self.n_rules))
            for r in range(self.n_rules):
                mf_values[:, r] = self._gaussian_mf(
                    X[:, i],
                    self.antecedent_params[i]['means'][r],
                    self.antecedent_params[i]['sigmas'][r]
                )
            firing_strengths *= mf_values  # 逐元素相乘

        return firing_strengths

    def _compute_gradients(self, X, y, firing_strengths):
        """计算参数梯度"""
        y_pred = self.predict(X)
        error = y_pred - y
        n_samples, n_features = X.shape
        gradients = []

        # 计算前件参数梯度
        for dim in range(n_features):
            dim_grad = {'means': np.zeros(self.n_rules),
                        'sigmas': np.zeros(self.n_rules)}

            for r in range(self.n_rules):
                x_dim = X[:, dim]
                mean_r = self.antecedent_params[dim]['means'][r]
                sigma_r = self.antecedent_params[dim]['sigmas'][r]

                # 计算隶属度对参数的偏导
                d_mu_d_mean = firing_strengths[:, r] * (x_dim - mean_r) / (sigma_r ** 2 + 1e-8)
                d_mu_d_sigma = firing_strengths[:, r] * ((x_dim - mean_r) ** 2) / (sigma_r ** 3 + 1e-8)

                # 计算误差梯度
                sum_firing = np.sum(firing_strengths, axis=1)
                consequent_r = np.dot(X, self.consequent_params[r, :-1].T) + self.consequent_params[r, -1]

                common_term = error * (consequent_r - y_pred) / (sum_firing + 1e-8)

                grad_mean = 2 * np.mean(common_term * d_mu_d_mean)
                grad_sigma = 2 * np.mean(common_term * d_mu_d_sigma)

                dim_grad['means'][r] = grad_mean
                dim_grad['sigmas'][r] = grad_sigma

            gradients.append(dim_grad)

        return gradients

    def _update_consequent(self, X, y, firing_strengths):
        """在线更新后件参数"""
        X_ext = np.hstack([X, np.ones((X.shape[0], 1))])
        sum_firing = np.sum(firing_strengths, axis=1, keepdims=True) + 1e-8

        # 计算每个规则的权重
        rule_weights = firing_strengths / sum_firing

        # 计算后件参数梯度
        y_pred = self.predict(X)
        error = (y_pred - y).reshape(-1, 1)

        for r in range(self.n_rules):
            grad = (error * rule_weights[:, r].reshape(-1, 1)) * X_ext
            avg_grad = np.mean(grad, axis=0)
            self.consequent_params[r] -= self.lr * avg_grad

    def partial_fit(self, X, y):
        """在线学习接口"""
        X = np.array(X)
        y = y.flatten()

        # 首次调用时完成初始化
        if self.antecedent_params is None or self.consequent_params is None:
            self._check_initialization(X)

        # 计算激活强度
        firing_strengths = self._compute_firing_strength(X)

        # 更新后件参数
        self._update_consequent(X, y, firing_strengths)

        # 更新前件参数
        gradients = self._compute_gradients(X, y, firing_strengths)
        for dim in range(X.shape[1]):
            self.antecedent_params[dim]['means'] -= self.lr * gradients[dim]['means']
            self.antecedent_params[dim]['sigmas'] -= self.lr * gradients[dim]['sigmas']

    def predict(self, X):
        """预测输出"""
        X = np.array(X)
        if self.antecedent_params is None or self.consequent_params is None:
            raise ValueError("Model not initialized. Call partial_fit first.")

        firing_strengths = self._compute_firing_strength(X)
        X_ext = np.hstack([X, np.ones((X.shape[0], 1))])

        # 计算各规则输出
        rule_outputs = np.dot(X_ext, self.consequent_params.T)

        # 加权平均
        sum_firing = np.sum(firing_strengths, axis=1, keepdims=True) + 1e-8
        return np.sum(firing_strengths * rule_outputs, axis=1) / sum_firing.squeeze()


# 示例使用
if __name__ == "__main__":
    # 生成数据
    X, y = make_regression(n_samples=1000, n_features=2, noise=0.1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 专家规则设置示例（可选）
    expert_antecedent = [
        {'means': [-2.0, 0.0, 2.0], 'sigmas': [1.0, 1.0, 1.0]},  # 特征1的隶属函数
        {'means': [-1.5, 0.0, 1.5], 'sigmas': [0.8, 0.8, 0.8]}  # 特征2的隶属函数
    ]
    expert_consequent = np.array([
        [0.5, -0.3, 1.0],  # 规则1的后件参数
        [-0.2, 0.4, 0.8],  # 规则2
        [0.3, 0.1, -0.5]  # 规则3
    ])

    # 初始化带专家规则的模型
    model = TSKFuzzySystem(
        n_rules=3,
        learning_rate=0.01,
        antecedent_params=expert_antecedent,
        consequent_params=expert_consequent
    )

    # 模拟在线学习过程
    batch_size = 100
    for i in range(0, len(X_train), batch_size):
        X_batch = X_train[i:i + batch_size]
        y_batch = y_train[i:i + batch_size]
        model.partial_fit(X_batch, y_batch)

        # 可选：监控过程损失
        y_pred = model.predict(X_batch)
        loss = mean_squared_error(y_batch, y_pred)
        print(f"Batch {i // batch_size}, Loss: {loss:.4f}")

    # 最终评估
    y_pred = model.predict(X_test)
    print(f"\nFinal Test MSE: {mean_squared_error(y_test, y_pred):.4f}")