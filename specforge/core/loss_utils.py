import numpy as np
import matplotlib.pyplot as plt

# 设置绘图风格
plt.style.use('ggplot')
plt.rcParams['font.sans-serif'] = ['SimHei'] # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False # 用来正常显示负号

# 1. 定义置信度范围 (0.0 到 1.0)
confidence = np.linspace(0, 1, 200)
threshold = 0.6

# 2. 定义参数列表
lambdas = [3, 5, 8]      # 指数衰减参数
ks = [5, 10, 20]         # Sigmoid 陡峭度参数

# 3. 创建画布
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# ==========================================
# 方案 A：指数衰减 (Exponential Decay)
# w(c) = exp(-lambda * (c - 0.6))
# ==========================================
ax1.set_title(r'方案 A：指数衰减 (Exponential Decay)', fontsize=15, pad=15)
ax1.set_xlabel('置信度 (Confidence)', fontsize=12)
ax1.set_ylabel('梯度权重 (Weight)', fontsize=12)

for lam in lambdas:
    # 计算权重
    # 当 c < 0.6 时，指数为正，权重 > 1 (梯度放大)
    # 当 c > 0.6 时，指数为负，权重 < 1 (梯度抑制)
    weights = np.exp(-lam * (confidence - threshold))
    ax1.plot(confidence, weights, label=f'$\lambda = {lam}$', linewidth=2)

# 辅助线
ax1.axvline(x=threshold, color='gray', linestyle='--', alpha=0.7)
ax1.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Baseline (Weight=1)')

# 标注关键区域
ax1.annotate(r'Conf < 0.6: 权重 > 1 (放大梯度)', xy=(0.2, 5), xytext=(0.4, 6),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))
ax1.annotate(r'Conf > 0.6: 权重 < 1 (抑制梯度)', xy=(0.8, 0.2), xytext=(0.65, 0.3),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))

ax1.legend()
ax1.grid(True, alpha=0.3)

# ==========================================
# 方案 B：Sigmoid 反转 (Sigmoid Reverse)
# w(c) = 1 / (1 + exp(k * (c - 0.6)))
# ==========================================
ax2.set_title(r'方案 B：Sigmoid 反转 (Sigmoid Reverse)', fontsize=15, pad=15)
ax2.set_xlabel('置信度 (Confidence)', fontsize=12)
ax2.set_ylabel('梯度权重 (Weight)', fontsize=12)

for k in ks:
    # 计算权重
    # 这是一个 S 型曲线，以 0.6 为中心反转
    # 注意：Sigmoid 的值域通常是 (0, 1)，所以权重最大不超过 1
    # 如果想要放大幅度，需要乘以一个系数 (例如 * 2.0)
    weights = 1.0 / (1.0 + np.exp(k * (confidence - threshold)))
    ax2.plot(confidence, weights, label=f'$k = {k}$', linewidth=2)

# 辅助线
ax2.axvline(x=threshold, color='gray', linestyle='--', alpha=0.7)
ax2.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Baseline (Weight=1)')

# 标注
ax2.annotate(r'Conf < 0.6: 权重趋近 1', xy=(0.2, 0.9), xytext=(0.4, 0.8),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))
ax2.annotate(r'Conf > 0.6: 权重趋近 0', xy=(0.8, 0.1), xytext=(0.65, 0.2),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))

ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("loss confidence.png")