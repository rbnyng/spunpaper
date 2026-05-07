import numpy as np
import matplotlib.pyplot as plt
import os

base_path = '/scratch/zf281/create_d-pixels_biomassters/data/test_agbm_representation'

file_name = '2018_0203607d_agbm.npy'

data = np.load(os.path.join(base_path, file_name)) # (H, W, C)
print(f'Data shape: {data.shape}')
# 取前三个波段可视化
data_rgb = data[:, :, :3]  # 假设前三个波段是RGB
# 归一化到0-1范围
for i in range(3):
    data_rgb[:, :, i] = (data_rgb[:, :, i] - np.min(data_rgb[:, :, i])) / (np.max(data_rgb[:, :, i]) - np.min(data_rgb[:, :, i]))
plt.figure(figsize=(6, 6))
plt.imshow(data_rgb)
plt.title('Representation Visualization')
plt.axis('off')
plt.tight_layout()
plt.savefig('/scratch/zf281/create_d-pixels_biomassters/representation.png', dpi=300)
plt.close()