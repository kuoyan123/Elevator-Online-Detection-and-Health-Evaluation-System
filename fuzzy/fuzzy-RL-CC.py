import numpy as np
from collections import deque
import random
from scipy.stats import linregress


class EnvironmentalRiskSubsystem:
    """环境风险模糊子系统（温度+湿度）"""

    def __init__(self):
        # 温度隶属函数（℃）
        self.temp_mf = [
            ('低温', 10, 3),
            ('常温', 25, 5),
            ('高温', 40, 3)
        ]

        # 湿度隶属函数（%RH）
        self.humid_mf = [
            ('干燥', 30, 8),
            ('适宜', 60, 10),
            ('潮湿', 90, 8)
        ]

        # 环境风险规则库
        self.rules = [
            (('低温', '干燥'), '好'),
            (('常温', '适宜'), '好'),
            (('高温', '潮湿'), '特别恶劣'),
            (('高温', '干燥'), '非常恶劣'),
            (('低温', '潮湿'), '较恶劣'),
        ]

        self.risk_levels = {'好': 0, '一般': 1, '较恶劣': 2, '非常恶劣': 3, '特别恶劣': 4}

    def _gauss(self, x, mean, sigma):
        return np.exp(-((x - mean) ** 2) / (2 * sigma ** 2))

    def compute_risk(self, temp, humid):
        """计算环境风险值（0-4连续值）"""
        # 计算隶属度
        temp_deg = {name: self._gauss(temp, m, s) for name, m, s in self.temp_mf}
        humid_deg = {name: self._gauss(humid, m, s) for name, m, s in self.humid_mf}

        # 规则推理
        risk_scores = np.zeros(5)
        for (t, h), r in self.rules:
            fire = min(temp_deg[t], humid_deg[h])
            risk_scores[self.risk_levels[r]] += fire

        # 去模糊化
        total = risk_scores.sum()
        return np.dot(risk_scores, np.arange(5)) / total if total > 0 else 2.0


class ConfidenceManager:
    """置信度管理模块"""

    def __init__(self, rule_count):
        self.rule_history = [deque(maxlen=100) for _ in range(rule_count)]
        self.confidence_trend = deque(maxlen=30)

    def update_rule(self, idx, output):
        self.rule_history[idx].append(output)

    def calculate(self, firing_strengths):
        """计算系统置信度（0-1）"""
        uncertainties = []
        for i, strength in enumerate(firing_strengths):
            data = list(self.rule_history[i])
            if len(data) < 10: continue
            sigma = np.std(data) * strength
            uncertainties.append(sigma)

        if not uncertainties: return 1.0
        total = sum(firing_strengths)
        return 1 - np.tanh(sum(uncertainties) / total)

    def analyze_trend(self):
        """分析置信度变化趋势"""
        if len(self.confidence_trend) < 10: return 0
        return linregress(range(len(self.confidence_trend)), list(self.confidence_trend)).slope


class ElevatorEnvEvaluator:
    """电梯环境评估主系统"""

    def __init__(self):
        # 子系统初始化
        self.env_risk = EnvironmentalRiskSubsystem()

        # 使用频次隶属函数（次/日）
        self.freq_mf = [
            ('极低', 75, 25),  # 均值75，标准差25
            ('较低', 200, 50),
            ('中等', 500, 100),
            ('较高', 1000, 200),
            ('很高', 2000, 300),
            ('极高', 3000, 400)
        ]

        # 专家规则（Mamdani型）
        self.expert_rules = [
            (('极低', '好'), 90),
            (('极高', '特别恶劣'), 10),
            (('中等', '较恶劣'), 65),
        ]

        # TSK自动规则
        self.tsk_rules = self._init_tsk_rules(10)

        # 学习系统
        self.conf_mgr = ConfidenceManager(len(self.expert_rules) + len(self.tsk_rules))
        self.reward_history = deque(maxlen=100)
        self.learning_rate = 0.02

    def _init_tsk_rules(self, num):
        """初始化TSK规则"""
        return [{
            'antecedent': (
                random.choice([m[0] for m in self.freq_mf]),
                random.choice(['好', '一般', '较恶劣', '非常恶劣', '特别恶劣'])
            ),
            'consequent': np.random.randn(3) * 0.1  # [w_freq, w_risk, bias]
        } for _ in range(num)]

    def _compute_firing(self, freq, risk_level):
        """计算隶属度和触发强度"""
        freq_deg = {name: self._gauss(freq, m, s) for name, m, s in self.freq_mf}
        risk_deg = self._gauss(risk_level, *{
            '好': (0, 0.5), '一般': (1, 0.5), '较恶劣': (2, 0.5),
            '非常恶劣': (3, 0.5), '特别恶劣': (4, 0.5)
        }[self._map_risk(risk_level)])
        return freq_deg, risk_deg

    def _map_risk(self, value):
        """将连续风险值映射到标签"""
        if value < 0.5:
            return '好'
        elif value < 1.5:
            return '一般'
        elif value < 2.5:
            return '较恶劣'
        elif value < 3.5:
            return '非常恶劣'
        else:
            return '特别恶劣'

    def _gauss(self, x, mean, sigma):
        return np.exp(-(x - mean) ** 2 / (2 * sigma ** 2))

    def predict(self, freq, temp, humid):
        """执行环境评估"""
        # 计算环境风险
        risk_level = self.env_risk.compute_risk(temp, humid)

        # 计算各规则触发强度
        freq_deg, risk_deg = self._compute_firing(freq, risk_level)
        firing_strengths = []
        outputs = []

        # 处理专家规则
        for i, ((f_level, r_level), score) in enumerate(self.expert_rules):
            fire = min(freq_deg[f_level], risk_deg)
            self.conf_mgr.update_rule(i, score)
            firing_strengths.append(fire)
            outputs.append(score)

        # 处理TSK规则
        for j, rule in enumerate(self.tsk_rules, start=len(self.expert_rules)):
            f_level, r_level = rule['antecedent']
            w1, w2, b = rule['consequent']
            fire = min(freq_deg[f_level], risk_deg)
            output = w1 * freq + w2 * risk_level + b

            self.conf_mgr.update_rule(j, output)
            firing_strengths.append(fire)
            outputs.append(output)

        # 综合评估
        total_fire = sum(firing_strengths)
        final_score = np.dot(firing_strengths, outputs) / total_fire if total_fire > 0 else 50

        # 计算置信度
        confidence = self.conf_mgr.calculate(firing_strengths)
        self.conf_mgr.confidence_trend.append(confidence)

        return {
            'score': final_score,
            'confidence': confidence,
            'risk_level': risk_level,
            'decision_level': self._decision_level(confidence)
        }

    def _decision_level(self, confidence):
        """确定决策等级"""
        trend = self.conf_mgr.analyze_trend()
        if confidence > 0.8 and trend >= -0.05:
            return 'auto'
        elif confidence > 0.6:
            return 'semi-auto'
        else:
            return 'manual'

    def update(self, actual_score):
        """强化学习更新"""
        self.reward_history.append(actual_score)

        # 计算趋势奖励
        if len(self.reward_history) >= 20:
            x = np.arange(len(self.reward_history))
            trend = linregress(x, self.reward_history).slope
            reward = -np.sign(trend) * np.log1p(abs(trend))
            self._adjust_rules(reward)

    def _adjust_rules(self, reward):
        """调整TSK规则参数"""
        for rule in self.tsk_rules:
            delta = reward * self.learning_rate
            rule['consequent'] += np.array([delta * 0.1, delta * 0.05, delta * 0.01])
            # 参数约束
            rule['consequent'] = np.clip(rule['consequent'], -1, 1)


# 模拟测试
# if __name__ == "__main__":
#     system = ElevatorEnvEvaluator()
#
#     # 测试用例
#     test_cases = [
#         (80, 18, 35),  # 低频+适宜环境
#         (2200, 42, 85)  # 高频+恶劣环境
#     ]
#
#     for i, (freq, temp, humid) in enumerate(test_cases):
#         result = system.predict(freq, temp, humid)
#         print(f"案例{i + 1}: {freq}次/日, {temp}℃, {humid}%RH")
#         print(f"环境风险: {result['risk_level']:.2f}")
#         print(f"综合评分: {result['score']:.1f}")
#         print(f"置信度: {result['confidence']:.2f} | 决策等级: {result['decision_level']}")
#         print("=" * 50)
#
#         # 模拟实际评分（假设首次评估偏高）
#         system.update(result['score'] - 10 if i == 0 else result['score'] + 5)
#
#     # 显示更新后的TSK规则
#     print("\n更新后的TSK规则示例:")
#     print("前件:", system.tsk_rules[0]['antecedent'])
#     print("参数:", system.tsk_rules[0]['consequent'].round(3))

class EnhancedElevatorEnvEvaluator(ElevatorEnvEvaluator):
    """增强版评估系统（支持输入缺失）"""

    def __init__(self):
        super().__init__()
        # 初始化默认值（根据历史数据统计）
        self.default_values = {
            'freq': 800,  # 平均日使用次数
            'temp': 25,  # 典型温度
            'humid': 60  # 典型湿度
        }
        self.missing_flags = set()

    def handle_missing(self, inputs):
        """处理缺失输入并记录标记"""
        self.missing_flags.clear()
        handled = {}

        for key in ['freq', 'temp', 'humid']:
            if inputs.get(key) is None:
                handled[key] = self.default_values[key]
                self.missing_flags.add(key)
            else:
                handled[key] = inputs[key]

        return handled

    def _compute_risk_with_missing(self, temp, humid):
        """处理环境风险计算的缺失情况"""
        if 'temp' in self.missing_flags:
            # 温度缺失时根据湿度推断
            if humid > 70:
                return 3.8  # 高湿度通常伴随高温
            elif humid < 40:
                return 1.2
            return 2.5
        elif 'humid' in self.missing_flags:
            # 湿度缺失时根据温度推断
            if temp > 35:
                return 3.5
            elif temp < 15:
                return 1.5
            return 2.0
        else:
            return self.env_risk.compute_risk(temp, humid)

    def _adjust_freq_mf(self, freq):
        """调整频次隶属函数应对缺失"""
        if 'freq' in self.missing_flags:
            # 缺失时扩大隶属函数范围
            return [(name, m, s * 2) for name, m, s in self.freq_mf]
        return self.freq_mf

    def predict(self, **inputs):
        """支持带缺失的预测"""
        # 处理缺失值
        valid_inputs = self.handle_missing(inputs)
        freq = valid_inputs['freq']
        temp = valid_inputs['temp']
        humid = valid_inputs['humid']

        # 计算环境风险（考虑缺失）
        risk_level = self._compute_risk_with_missing(temp, humid)

        # 调整频次隶属函数
        adjusted_mf = self._adjust_freq_mf(freq)
        freq_deg = {name: self._gauss(freq, m, s) for name, m, s in adjusted_mf}

        # 风险隶属度计算
        risk_deg = self._gauss(risk_level, *{
            '好': (0, 0.7), '一般': (1, 0.7), '较恶劣': (2, 0.7),
            '非常恶劣': (3, 0.7), '特别恶劣': (4, 0.7)
        }[self._map_risk(risk_level)])

        # 执行模糊推理
        firing_strengths = []
        outputs = []

        # 专家规则处理
        for (f_level, r_level), score in self.expert_rules:
            fire = min(freq_deg[f_level], risk_deg)
            firing_strengths.append(fire)
            outputs.append(score)

        # TSK规则处理
        for rule in self.tsk_rules:
            f_level, r_level = rule['antecedent']
            w1, w2, b = rule['consequent']
            fire = min(freq_deg[f_level], risk_deg)
            output = w1 * freq + w2 * risk_level + b
            firing_strengths.append(fire)
            outputs.append(output)

        # 综合评估
        total_fire = sum(firing_strengths)
        final_score = np.dot(firing_strengths, outputs) / total_fire if total_fire > 0 else 50

        # 调整置信度
        base_confidence = self.conf_mgr.calculate(firing_strengths)
        adjusted_confidence = base_confidence * (0.8 ** len(self.missing_flags))

        return {
            'score': final_score,
            'confidence': adjusted_confidence,
            'missing': list(self.missing_flags),
            'decision_level': self._decision_level(adjusted_confidence)
        }


# 测试用例
if __name__ == "__main__":
    system = EnhancedElevatorEnvEvaluator()

    # 完整输入测试
    print("测试1：完整输入")
    res1 = system.predict(freq=1200, temp=28, humid=65)
    print(f"评分: {res1['score']:.1f} 置信度: {res1['confidence']:.2f} 缺失: {res1['missing']}")

    # 温度缺失测试
    print("\n测试2：温度缺失")
    res2 = system.predict(freq=800, temp=None, humid=75)
    print(f"评分: {res2['score']:.1f} 置信度: {res2['confidence']:.2f} 缺失: {res2['missing']}")

    # 频次缺失测试
    print("\n测试3：频次缺失")
    res3 = system.predict(freq=None, temp=38, humid=80)
    print(f"评分: {res3['score']:.1f} 置信度: {res3['confidence']:.2f} 缺失: {res3['missing']}")

    # 多输入缺失测试
    print("\n测试4：温湿度缺失")
    res4 = system.predict(freq=2000, temp=None, humid=None)
    print(f"评分: {res4['score']:.1f} 置信度: {res4['confidence']:.2f} 缺失: {res4['missing']}")