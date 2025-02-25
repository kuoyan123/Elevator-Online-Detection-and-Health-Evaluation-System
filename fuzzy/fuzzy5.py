import numpy as np
from collections import deque
import random


class OnlineFIS:
    """在线模糊推理系统（强化学习版本）"""

    def __init__(self, n_features, expert_rules, auto_rule_count=3):
        self.n_features = n_features
        self.memory = deque(maxlen=1000)  # 经验回放缓冲区

        # 初始化模糊系统组件
        self._init_membership_functions()
        self._init_rules(expert_rules, auto_rule_count)

        # 学习参数
        self.learning_rate = 0.01
        self.exploration_rate = 0.2
        self.batch_size = 32

    def _init_membership_functions(self):
        """初始化隶属函数（每个特征3个：低、中、高）"""
        self.mfs = []
        for _ in range(self.n_features):
            self.mfs.append([
                {'mean': -2.0, 'sigma': 1.0},
                {'mean': 0.0, 'sigma': 1.0},
                {'mean': 2.0, 'sigma': 1.0}
            ])

    def _init_rules(self, expert_rules, auto_rule_count):
        """初始化规则库"""
        self.rules = []
        # 专家规则（Mamdani型）
        for indices, output in expert_rules:
            self.rules.append({
                'type': 'mamdani',
                'antecedent': indices,
                'consequent': output
            })

        # 自动规则（TSK型）
        for _ in range(auto_rule_count):
            antecedent = [random.randint(0, 2) for _ in range(self.n_features)]
            self.rules.append({
                'type': 'tsk',
                'antecedent': antecedent,
                'consequent': np.random.randn(self.n_features + 1) * 0.1
            })

    def compute_firing(self, x):
        """计算规则触发强度"""
        firing = np.ones(len(self.rules))
        for i, rule in enumerate(self.rules):
            for feat, mf_idx in enumerate(rule['antecedent']):
                mf = self.mfs[feat][mf_idx]
                x_val = x[feat]
                firing[i] *= np.exp(-(x_val - mf['mean']) ** 2 / (2 * mf['sigma'] ** 2))
        return firing

    def predict(self, x):
        """执行推理"""
        firing = self.compute_firing(x)
        outputs = []

        for i, rule in enumerate(self.rules):
            if rule['type'] == 'mamdani':
                outputs.append(rule['consequent'])
            else:
                outputs.append(np.dot(x, rule['consequent'][:-1]) + rule['consequent'][-1])

        total_firing = np.sum(firing)
        if total_firing < 1e-6:
            return 0.0
        return np.dot(firing, outputs) / total_firing

    def _get_current_state(self):
        """获取当前参数状态"""
        state = []
        # 隶属函数参数
        for feat in self.mfs:
            for mf in feat:
                state.extend([mf['mean'], mf['sigma']])
        # TSK规则参数
        for rule in self.rules:
            if rule['type'] == 'tsk':
                state.extend(rule['consequent'])
        return np.array(state)

    def _apply_action(self, action):
        """应用参数调整动作"""
        ptr = 0
        # 更新隶属函数
        for feat in self.mfs:
            for mf in feat:
                mf['mean'] += action[ptr] * 0.1  # 调整幅度缩放
                mf['sigma'] = np.clip(mf['sigma'] + action[ptr + 1] * 0.01, 0.5, 2.0)
                ptr += 2
        # 更新TSK参数
        for rule in self.rules:
            if rule['type'] == 'tsk':
                size = len(rule['consequent'])
                rule['consequent'] += action[ptr:ptr + size] * 0.01
                ptr += size

    def _exploration_noise(self):
        """生成探索噪声"""
        state_size = len(self._get_current_state())
        return np.random.normal(0, self.exploration_rate, state_size)

    def store_experience(self, x, error_info):
        """存储经验数据"""
        self.memory.append((x, error_info))

    def update_policy(self):
        """执行策略更新"""
        if len(self.memory) < self.batch_size:
            return 0.0  # 返回默认值

        batch = random.sample(self.memory, self.batch_size)
        total_reward = 0

        for x, (error_flag, error_value) in batch:
            # 当前状态
            current_state = self._get_current_state()

            # 生成动作（参数调整方向）
            action = self._exploration_noise()

            # 应用动作前的预测
            prev_pred = self.predict(x)

            # 应用动作
            self._apply_action(action)

            # 计算奖励
            new_pred = self.predict(x)
            reward = self._calculate_reward(prev_pred, new_pred, error_flag, error_value)
            total_reward += reward

            # 保留有效更新或回滚
            if reward > 0:
                self._apply_action(action)  # 确认更新
            else:
                self._apply_action(-action)  # 回滚

        # 动态调整探索率
        self.exploration_rate *= 0.99
        return total_reward / self.batch_size

    def _calculate_reward(self, prev_pred, new_pred, error_flag, error_value):
        """计算奖励值（基于系统输出）"""
        # 错误检测基于预测值
        target = 0.0  # 假设理想输出为0
        prev_error = abs(prev_pred - target)
        new_error = abs(new_pred - target)

        # 奖励改进程度
        improvement = prev_error - new_error
        return np.tanh(improvement * error_value)

    def online_update(self, x):
        """在线更新入口"""
        # 1. 执行预测
        prediction = self.predict(x)

        # 2. 基于预测结果检测错误（示例逻辑）
        error_flag, error_value = self._error_detector(prediction)

        # 3. 存储当前状态
        self.store_experience(x, (error_flag, error_value))

        # 4. 执行策略更新
        avg_reward = self.update_policy()
        return avg_reward, prediction

    def _error_detector(self, prediction):
        """内置错误检测器（示例实现）"""
        threshold = 1.0
        error_magnitude = abs(prediction)

        if error_magnitude > threshold:
            error_value = error_magnitude - threshold
            return ('偏大', error_value)
        else:
            return ('正常', 0.0)


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


#
# # 使用示例
# if __name__ == "__main__":
#     # 初始化专家规则
#     expert_rules = [
#         ([0, 0], -1.0),  # 特征低+低 → 正常
#         ([2, 2], 1.0)  # 特征高+高 → 异常
#     ]
#
#     # 创建在线学习系统
#     fis = OnlineFIS(n_features=2, expert_rules=expert_rules)
#
#     # 模拟在线学习过程
#     for step in range(1000):
#         # 模拟实时数据输入（特征值）
#         x = np.random.randn(2) * 2
#
#         # 执行在线更新
#         avg_reward, prediction = fis.online_update(x)
#
#         # 监控学习过程
#         if step % 100 == 0:
#             print(f"Step {step}: Pred={prediction:.2f} Avg Reward={avg_reward:.2f} Explore={fis.exploration_rate:.3f}")
#
#     # 验证最终参数
#     print("\n专家规则保持:")
#     for rule in fis.rules[:2]:
#         print(f"Rule: {rule['antecedent']} → {rule['consequent']}")
#
#     print("\n自动规则参数示例:")
#     print(fis.rules[2]['consequent'])
#
#     x = np.linspace(-2, 2, 100)
#     y = np.linspace(0, 2, 100)
#     X, Y = np.meshgrid(x, y)
#     Z = np.tanh((np.abs(X) - np.abs(Y)) * (np.abs(X) - 1.0))
#
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     ax.plot_surface(X, Y, Z)
#     ax.set_xlabel('Previous Error')
#     ax.set_ylabel('New Error')
#     ax.set_zlabel('Reward')
#     plt.show()
import numpy as np
from collections import deque
import random
from scipy.stats import gaussian_kde


class DynamicTargetManager:
    """动态目标值管理模块"""

    def __init__(self, init_target=0.0):
        self.target_history = deque(maxlen=500)  # 历史目标记录
        self.current_target = init_target
        self.target_adjustment_step = 0.1  # 目标调整步长

    def update_target(self, new_target):
        """直接设置新目标值"""
        self.target_history.append(new_target)
        self.current_target = new_target

    def adaptive_adjust(self, predictions):
        """基于预测分布的自动调整"""
        if len(predictions) < 100:
            return  # 积累足够数据前不调整

        # 计算预测值概率密度
        kde = gaussian_kde(predictions)
        x = np.linspace(min(predictions), max(predictions), 100)
        densities = kde(x)

        # 寻找主峰位置作为新目标
        main_peak = x[np.argmax(densities)]
        self.current_target = 0.9 * self.current_target + 0.1 * main_peak

    def get_target(self):
        return self.current_target


class EnhancedOnlineFIS(OnlineFIS):
    """增强型在线模糊系统"""

    def __init__(self, n_features, expert_rules,
                 auto_rule_count=3,
                 dynamic_target=True):
        super().__init__(n_features, expert_rules, auto_rule_count)
        self.dynamic_target = dynamic_target
        self.target_manager = DynamicTargetManager()
        self.prediction_history = deque(maxlen=1000)

    def _calculate_reward(self, prev_pred, new_pred, error_flag, error_value):
        """基于动态目标的奖励计算"""
        target = self.target_manager.get_target()

        prev_error = abs(prev_pred - target)
        new_error = abs(new_pred - target)
        improvement = prev_error - new_error

        # 引入误差方向敏感性
        direction_factor = 1.0
        if (new_pred > target and error_flag == '偏大') or \
                (new_pred < target and error_flag == '偏小'):
            direction_factor = 1.5

        return np.tanh(improvement * error_value * direction_factor)

    def online_update(self, x, external_target=None):
        """增强型在线更新"""
        # 执行预测
        prediction = self.predict(x)
        self.prediction_history.append(prediction)

        # 更新目标值
        if self.dynamic_target:
            if external_target is not None:
                self.target_manager.update_target(external_target)
            else:
                self.target_manager.adaptive_adjust(self.prediction_history)

        # 错误检测
        target = self.target_manager.get_target()
        error = prediction - target
        error_magnitude = abs(error)

        if error_magnitude > 1.0:  # 动态阈值
            error_flag = '偏大' if error > 0 else '偏小'
            error_value = error_magnitude - 1.0
        else:
            error_flag = '正常'
            error_value = 0.0

        # 存储经验
        self.store_experience(x, (error_flag, error_value))

        # 策略更新
        avg_reward = self.update_policy()

        return avg_reward, prediction, target


# 使用示例
if __name__ == "__main__":
    # 初始化系统（电梯健康监测场景）
    expert_rules = [
        ([0, 0], -1.0),  # 振动小+噪音小 → 健康
        ([2, 2], 1.0)  # 振动大+噪音大 → 故障
    ]

    fis = EnhancedOnlineFIS(
        n_features=2,
        expert_rules=expert_rules,
        dynamic_target=True
    )

    # 模拟电梯运行数据（分阶段场景）
    phases = [
        {'health': 0.9, 'duration': 300},  # 健康阶段
        {'health': 0.6, 'duration': 400},  # 性能下降
        {'health': 0.3, 'duration': 300}  # 故障前期
    ]

    current_phase = 0
    phase_counter = 0

    for step in range(1000):
        # 阶段转换检测
        if phase_counter >= phases[current_phase]['duration']:
            current_phase = min(current_phase + 1, len(phases) - 1)
            phase_counter = 0

        # 生成模拟数据
        health = phases[current_phase]['health']
        x = np.random.randn(2) * (2 - health) + health * 1.5

        # 执行在线更新
        avg_reward, pred, target = fis.online_update(x)

        # 监控输出
        if step % 50 == 0:
            print(f"Step {step:04d} | Target: {target:.2f} | Pred: {pred:.2f} | Reward: {avg_reward:.2f}")

        phase_counter += 1

    # 输出最终状态
    print("\n动态目标演化:")
    print(f"初始目标: 0.00")
    print(f"最终目标: {fis.target_manager.get_target():.2f}")

    print("\n专家规则保持:")
    for rule in fis.rules[:2]:
        print(f"Rule: {rule['antecedent']} → {rule['consequent']}")

    print("\n自动规则参数示例:")
    print(fis.rules[2]['consequent'])