from PIL import Image
import numpy as np
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

# 加载训练数据
digits = datasets.load_digits()
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

# 创建并训练模型
model = RandomForestClassifier(n_estimators=10)
model.fit(data[:n_samples // 2], digits.target[:n_samples // 2])

# 加载并预处理输入图像
img = Image.open('mlzukan-img.png').convert('L')
img_resized = img.resize((8, 8), Image.Resampling.LANCZOS)
img_array = np.array(img_resized)
img_array = (16 - img_array * 16 / 255).astype(int)  # 或 np.int64, np.int32 根据需要

# 使用模型进行预测
img_data = img_array.reshape(1, -1)
predicted_digit = model.predict(img_data)

print("Predicted digit:", predicted_digit[0])
