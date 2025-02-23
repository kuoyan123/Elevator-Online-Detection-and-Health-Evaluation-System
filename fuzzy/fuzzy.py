import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


class TSKFuzzySystem:
    def __init__(self, n_rules=5, learning_rate=0.01, epochs=100):
        self.n_rules = n_rules
        self.lr = learning_rate
        self.epochs = epochs
        self.antecedent_params = []  # 前件参数：每个特征的均值和标准差
        self.consequent_params = None  # 后件参数矩阵 (n_rules, n_features+1)

    def _init_membership_functions(self, X):
        """初始化前件参数（高斯隶属函数的均值和标准差）"""
        n_features = X.shape[1]
        self.antecedent_params = []
        for i in range(n_features):
            min_val = np.min(X[:, i])
            max_val = np.max(X[:, i])
            means = np.linspace(min_val, max_val, self.n_rules)
            sigmas = np.ones(self.n_rules) * (max_val - min_val) / (2 * self.n_rules)
            self.antecedent_params.append({'means': means, 'sigmas': sigmas})

    def _gaussian_mf(self, x, mean, sigma):
        """高斯隶属函数"""
        return np.exp(-((x - mean) ** 2) / (2 * sigma ** 2 + 1e-8))

    def _compute_firing_strength(self, X):
        """计算每条规则的激活强度（隶属度的乘积）"""
        n_samples, n_features = X.shape
        firing_strengths = np.ones((n_samples, self.n_rules))
        for i in range(n_features):
            mf_values = np.zeros((n_samples, self.n_rules))
            for r in range(self.n_rules):
                mf_values[:, r] = self._gaussian_mf(X[:, i],
                                                    self.antecedent_params[i]['means'][r],
                                                    self.antecedent_params[i]['sigmas'][r])
            firing_strengths *= mf_values  # 逐元素相乘
        return firing_strengths

    def _consequent_function(self, X):
        """计算后件函数的输出（各规则的线性组合）"""
        X_ext = np.hstack([X, np.ones((X.shape[0], 1))])
        if self.consequent_params is None:
            # 初始化后件参数为零矩阵
            self.consequent_params = np.zeros((self.n_rules, X_ext.shape[1]))
        return X_ext.dot(self.consequent_params.T)  # (n_samples, n_rules)

    def _update_consequent(self, X, y, firing_strengths):
        """使用最小二乘法更新后件参数"""
        X_ext = np.hstack([X, np.ones((X.shape[0], 1))])
        # 构造设计矩阵Phi: (n_samples, n_rules * (n_features+1))
        Phi = np.zeros((X.shape[0], self.n_rules * X_ext.shape[1]))
        for r in range(self.n_rules):
            start = r * X_ext.shape[1]
            end = (r + 1) * X_ext.shape[1]
            Phi[:, start:end] = firing_strengths[:, r][:, np.newaxis] * X_ext
        # 最小二乘解
        self.consequent_params = np.linalg.lstsq(Phi, y, rcond=None)[0].reshape(self.n_rules, X_ext.shape[1])

    def _compute_gradients(self, X, y, firing_strengths):
        """计算前件参数（均值和标准差）的梯度"""
        y_pred = self.predict(X)
        error = y_pred - y  # 预测误差
        n_samples, n_features = X.shape
        gradients = []

        for dim in range(n_features):
            dim_grad = {'means': np.zeros(self.n_rules), 'sigmas': np.zeros(self.n_rules)}
            for r in range(self.n_rules):
                # 计算当前规则r对当前特征dim的梯度
                x_dim = X[:, dim]
                mean_r = self.antecedent_params[dim]['means'][r]
                sigma_r = self.antecedent_params[dim]['sigmas'][r]

                # 计算firing_strength对均值的导数
                d_mu_d_mean = firing_strengths[:, r] * (x_dim - mean_r) / (sigma_r ** 2 + 1e-8)
                # 计算firing_strength对标准差的导数
                d_mu_d_sigma = firing_strengths[:, r] * ((x_dim - mean_r) ** 2) / (sigma_r ** 3 + 1e-8)

                # 计算模型输出对firing_strength的导数
                sum_firing = np.sum(firing_strengths, axis=1)
                consequent_r = self._consequent_function(X)[:, r]
                common_term = (consequent_r - y_pred) / (sum_firing + 1e-8)

                # 合并梯度
                grad_mean = 2 * np.mean(error * common_term * d_mu_d_mean)
                grad_sigma = 2 * np.mean(error * common_term * d_mu_d_sigma)

                dim_grad['means'][r] = grad_mean
                dim_grad['sigmas'][r] = grad_sigma
            gradients.append(dim_grad)
        return gradients

    def fit(self, X, y):
        X = np.array(X)
        y = y.flatten()
        self._init_membership_functions(X)

        for epoch in range(self.epochs):
            firing_strengths = self._compute_firing_strength(X)
            self._update_consequent(X, y, firing_strengths)

            # 计算损失
            y_pred = self.predict(X)
            loss = mean_squared_error(y, y_pred)

            # 更新前件参数
            gradients = self._compute_gradients(X, y, firing_strengths)
            for dim in range(X.shape[1]):
                self.antecedent_params[dim]['means'] -= self.lr * gradients[dim]['means']
                self.antecedent_params[dim]['sigmas'] -= self.lr * gradients[dim]['sigmas']

            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

    def predict(self, X):
        X = np.array(X)
        firing_strengths = self._compute_firing_strength(X)
        consequent_outputs = self._consequent_function(X)
        sum_firing = np.sum(firing_strengths, axis=1, keepdims=True)
        sum_firing[sum_firing == 0] = 1e-8  # 防止除零
        y_pred = np.sum(firing_strengths * consequent_outputs, axis=1) / sum_firing.squeeze()
        return y_pred


# 示例使用
if __name__ == "__main__":
    # 生成回归数据
    X, y = make_regression(n_samples=1000, n_features=2, noise=0.1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 初始化并训练模型
    model = TSKFuzzySystem(n_rules=5, learning_rate=0.01, epochs=100)
    model.fit(X_train, y_train)

    # 评估模型
    y_pred = model.predict(X_test)
    print(f"Test MSE: {mean_squared_error(y_test, y_pred):.4f}")