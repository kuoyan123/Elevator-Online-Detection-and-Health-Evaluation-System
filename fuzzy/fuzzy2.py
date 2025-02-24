import numpy as np
from scipy.signal import lfilter
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 专家规则定义 (3个规则，每个特征3个隶属函数)
expert_antecedent = [
    {'means': [-2.0, 0.0, 2.0], 'sigmas': [1.0, 1.0, 1.0]},  # 特征1
    {'means': [-1.5, 0.0, 1.5], 'sigmas': [0.8, 0.8, 0.8]}  # 特征2
]

expert_consequent = np.array([
    [0.5, -0.3, 1.0],  # 规则1
    [-0.2, 0.4, 0.8],  # 规则2
    [0.3, 0.1, -0.5]  # 规则3
])


def generate_operational_data(n_samples=100):
    """生成模拟工况数据"""
    X = np.random.randn(n_samples, 2) * 2.0
    y = 3 * X[:, 0] + 2 * X[:, 1] + np.random.normal(0, 0.5, n_samples)
    return X, y


class TSKFuzzySystem:
    def __init__(self, n_rules=3, learning_rate=0.01,  # 默认规则数改为3
                 antecedent_params=None, consequent_params=None):
        self.n_rules = n_rules
        self.lr = learning_rate
        self.antecedent_params = antecedent_params
        self.consequent_params = consequent_params
        self.n_features = None

    def _check_initialization(self, X):
        if self.antecedent_params is None:
            self._init_membership_functions(X)
        if self.consequent_params is None:
            self._init_consequent_params(X)
        if self.n_features is None:
            self.n_features = X.shape[1]

    def _init_membership_functions(self, X):
        n_features = X.shape[1]
        self.antecedent_params = []
        for i in range(n_features):
            min_val = np.min(X[:, i])
            max_val = np.max(X[:, i])
            means = np.linspace(min_val, max_val, self.n_rules)
            sigmas = np.ones(self.n_rules) * (max_val - min_val) / (2 * self.n_rules)
            self.antecedent_params.append({'means': means, 'sigmas': sigmas})

    def _init_consequent_params(self, X):
        n_features = X.shape[1]
        self.consequent_params = np.random.randn(self.n_rules, n_features + 1) * 0.1

    def _gaussian_mf(self, x, mean, sigma):
        return np.exp(-((x - mean) ** 2) / (2 * sigma ** 2 + 1e-8))

    def _compute_firing_strength(self, X):
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
            firing_strengths *= mf_values
        return firing_strengths

    def _compute_gradients(self, X, y, firing_strengths):
        y_pred = self.predict(X)
        error = y_pred - y
        n_samples, n_features = X.shape
        gradients = []

        for dim in range(n_features):
            dim_grad = {'means': np.zeros(self.n_rules),
                        'sigmas': np.zeros(self.n_rules)}

            for r in range(self.n_rules):
                x_dim = X[:, dim]
                mean_r = self.antecedent_params[dim]['means'][r]
                sigma_r = self.antecedent_params[dim]['sigmas'][r]

                d_mu_d_mean = firing_strengths[:, r] * (x_dim - mean_r) / (sigma_r ** 2 + 1e-8)
                d_mu_d_sigma = firing_strengths[:, r] * ((x_dim - mean_r) ** 2) / (sigma_r ** 3 + 1e-8)

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
        X_ext = np.hstack([X, np.ones((X.shape[0], 1))])
        sum_firing = np.sum(firing_strengths, axis=1, keepdims=True) + 1e-8

        rule_weights = firing_strengths / sum_firing
        y_pred = self.predict(X)
        error = (y_pred - y).reshape(-1, 1)

        for r in range(self.n_rules):
            grad = (error * rule_weights[:, r].reshape(-1, 1)) * X_ext
            avg_grad = np.mean(grad, axis=0)
            self.consequent_params[r] -= self.lr * avg_grad

    def partial_fit(self, X, y):
        X = np.array(X)
        y = y.flatten()

        if self.antecedent_params is None or self.consequent_params is None:
            self._check_initialization(X)

        firing_strengths = self._compute_firing_strength(X)
        self._update_consequent(X, y, firing_strengths)

        gradients = self._compute_gradients(X, y, firing_strengths)
        for dim in range(X.shape[1]):
            self.antecedent_params[dim]['means'] -= self.lr * gradients[dim]['means']
            self.antecedent_params[dim]['sigmas'] -= self.lr * gradients[dim]['sigmas']

    def predict(self, X):
        X = np.array(X)
        if self.antecedent_params is None or self.consequent_params is None:
            raise ValueError("Model not initialized. Call partial_fit first.")

        firing_strengths = self._compute_firing_strength(X)
        X_ext = np.hstack([X, np.ones((X.shape[0], 1))])

        rule_outputs = np.dot(X_ext, self.consequent_params.T)
        sum_firing = np.sum(firing_strengths, axis=1, keepdims=True) + 1e-8
        return np.sum(firing_strengths * rule_outputs, axis=1) / sum_firing.squeeze()


class RLEnhancedTSK(TSKFuzzySystem):
    def __init__(self, n_rules=3, learning_rate=0.01,  # 规则数改为3
                 rule_priority=None, trend_window=10,
                 **kwargs):
        super().__init__(n_rules, learning_rate, **kwargs)

        self.trend_window = trend_window
        self.error_buffer = []
        self.trend_coeff = np.linspace(1.0, 0.5, trend_window)
        # self.rule_priority = rule_priority if rule_priority else np.ones(n_rules)
        self.rule_priority = rule_priority if rule_priority is not None else np.ones(n_rules)
        self.expert_rule_mask = (self.rule_priority == 0)

    def _calculate_trend_reward(self, current_errors):
        if len(self.error_buffer) < self.trend_window:
            return 0.0

        window_errors = np.array(self.error_buffer[-self.trend_window:])
        trend = lfilter(self.trend_coeff, 1.0, window_errors)[-1]

        return np.exp(-trend) - 1 if trend < 0 else -np.abs(trend)

    def _adaptive_learning_rate(self, reward):
        self.lr = np.clip(self.lr * (1 + 0.1 * reward), 1e-5, 0.1)

    def _apply_rule_constraints(self, gradients):
        for dim in range(len(gradients)):
            gradients[dim]['means'][self.expert_rule_mask] *= 0.1
            gradients[dim]['sigmas'][self.expert_rule_mask] *= 0.1
        return gradients

    def partial_fit(self, X, y, evaluate=False):
        y_pred = self.predict(X)
        super().partial_fit(X, y)

        if evaluate:
            current_errors = np.abs(y - y_pred).mean()
            self.error_buffer.append(current_errors)
            reward = self._calculate_trend_reward(current_errors)

            self._adaptive_learning_rate(reward)
            self.antecedent_params = self._apply_rule_constraints(self.antecedent_params)
            self.consequent_params[self.expert_rule_mask] *= 0.9

        return reward if evaluate else None


# if __name__ == "__main__":
#     # 修正参数匹配：3个规则，前2个为专家规则
#     rule_priority = np.array([0, 0, 1])  # 匹配3个规则
#     model = RLEnhancedTSK(
#         n_rules=3,
#         rule_priority=rule_priority,
#         antecedent_params=expert_antecedent,
#         consequent_params=expert_consequent
#     )
#
#     # 训练过程
#     for epoch in range(5):
#         X_batch, y_batch = generate_operational_data(1)  # 明确指定样本数量
#         reward = model.partial_fit(X_batch, y_batch, evaluate=True)
#         print(f"X_batch={X_batch}")
#         print(f"y_batch={y_batch}")
#
#         # 动态调整策略
#         if reward > 0.5:
#             model.rule_priority[2:] *= 0.95
#         elif reward < -0.2:
#             model.rule_priority[2:] = np.minimum(model.rule_priority[2:] * 1.1, 1.0)
#
#         print(f"Epoch {epoch:02d} | Reward: {reward:7.2f} | LR: {model.lr:.5f}")
#
#     # 最终测试
#     X_test, y_test = generate_operational_data(100)
#     y_pred = model.predict(X_test)
#     print(f"\nFinal MSE: {mean_squared_error(y_test, y_pred):.4f}")
if __name__ == "__main__":
    # 修正参数匹配：3个规则，前2个为专家规则
    rule_priority = np.array([0, 0, 1])  # 匹配3个规则
    model = RLEnhancedTSK(
        n_rules=3,
        rule_priority=rule_priority,
        antecedent_params=expert_antecedent,
        consequent_params=expert_consequent
    )

    # 输入数据
    input_data = np.array([[10, 10]])

    # 进行预测
    output = model.predict(input_data)

    print(f"输入 {input_data} 的输出结果是: {output[0]}")