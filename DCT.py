import numpy as np
import cv2
import matplotlib.pyplot as plt

# DCT将图像从时域转换到频域，用于图像压缩、图像特征提取

gray_img = cv2.imread('img.png', 0)
# DCT变换后图像通常包含大量的高频系数，需要对这些系数进行量化，以减少数据量
dct_img = cv2.dct(np.float32(gray_img))

# 提取感兴趣的特征区域（例如选取前n个系数）
n = 10  # 选择前10个系数作为特征
dct_feature = dct_img[:n, :n].flatten()
print(dct_feature)

# 量化DCT系数
quant_img = np.round(dct_img / 10) * 10
# 反量化DCT系数
dequant_img = quant_img * 10
# 反DCT变换
restore_img = cv2.idct(np.float32(dequant_img))

plt.subplot(1, 3, 1)
plt.imshow(gray_img, cmap='gray')
plt.title('Original Image')
plt.subplot(1, 3, 2)
plt.imshow(restore_img, cmap='gray')
plt.title('Restored Image')
plt.subplot(1, 3, 3)
plt.imshow(dct_img, cmap='gray')
plt.title('DCT Image')
plt.show()



