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

        # 记录触发的规则
        expert_rules = []
        tsk_rules = []

        # 处理专家规则
        for (f_level, r_level), score in self.expert_rules:
            fire = min(freq_deg[f_level], risk_deg)
            expert_rules.append({
                'desc': f"IF 使用频次是{f_level} AND 环境风险是{r_level} THEN 固定值={score}",
                'fire': fire,
                'output': score
            })

        # 处理TSK规则
        for rule in self.tsk_rules:
            f_level, r_level = rule['antecedent']
            w1, w2, b = rule['consequent']
            fire = min(freq_deg[f_level], risk_deg)
            output = w1 * freq + w2 * risk_level + b
            tsk_rules.append({
                'desc': f"IF 使用频次是{f_level} AND 环境风险是{r_level} THEN {w1:.2f}*freq + {w2:.2f}*risk + {b:.2f}",
                'fire': fire,
                'output': output
            })

        # 综合评估
        firing_strengths = [r['fire'] for r in expert_rules + tsk_rules]
        outputs = [r['output'] for r in expert_rules + tsk_rules]

        total_fire = sum(firing_strengths)
        final_score = np.dot(firing_strengths, outputs) / total_fire if total_fire > 0 else 50

        # 调整置信度
        base_confidence = self.conf_mgr.calculate(firing_strengths)
        adjusted_confidence = base_confidence * (0.8 ** len(self.missing_flags))

        return {
            'score': final_score,
            'confidence': adjusted_confidence,
            'missing': list(self.missing_flags),
            'expert_rules': expert_rules,
            'tsk_rules': tsk_rules,
            'decision_level': self._decision_level(adjusted_confidence)
        }

    def _decision_level(self, confidence):
        """返回更详细的决策等级说明"""
        trend = self.conf_mgr.analyze_trend()
        if confidence > 0.8 and trend >= -0.05:
            return '系统自动决策（置信度高且趋势稳定）'
        elif confidence > 0.6:
            return '半自动决策（需人工复核）'
        else:
            return '需完全人工决策（置信度不足）'


def print_rules(rules, title):
    """格式化打印规则触发情况"""
    print(f"\n{title}（触发强度>0.01的规则）:")
    for idx, rule in enumerate(rules, 1):
        if rule['fire'] > 0.01:
            print(f"规则{idx}: {rule['desc']}")
            print(f"   触发强度: {rule['fire']:.3f} | 输出值: {rule['output']:.1f}")


if __name__ == "__main__":
    system = EnhancedElevatorEnvEvaluator()

    test_cases = [
        {'freq': 1200, 'temp': 28, 'humid': 65},
        {'freq': 800, 'temp': None, 'humid': 75},
        {'freq': None, 'temp': 38, 'humid': 80},
        {'freq': 2000, 'temp': None, 'humid': None}
    ]

    for i, inputs in enumerate(test_cases, 1):
        print(f"\n{'=' * 50}\n测试案例 {i}: 输入参数={inputs}")
        result = system.predict(**inputs)

        # 打印基础结果
        print(f"\n评估结果:")
        print(f"综合评分: {result['score']:.1f}")
        print(f"置信度: {result['confidence']:.2f}")
        print(f"缺失字段: {result['missing'] or '无'}")
        print(f"决策等级: {result['decision_level']}")

        # 打印触发规则详情
        print_rules(result['expert_rules'], "专家规则")
        print_rules(result['tsk_rules'], "TSK自适应规则")

    # 显示决策等级说明
    print("\n决策等级说明:")
    print("1. 系统自动决策 - 置信度>0.8且趋势稳定，可直接执行自动控制")
    print("2. 半自动决策 - 置信度>0.6，建议系统推荐方案+人工确认")
    print("3. 需人工决策 - 置信度不足，需要完全人工判断")