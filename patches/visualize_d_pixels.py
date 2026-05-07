import numpy as np
import matplotlib.pyplot as plt
import os

base_path = '/scratch/zf281/create_d-pixels_biomassters/data/test_agbm_d-pixel/2020_2d090d13_agbm/data_processed'

### S1 data
s1_acending = f'{base_path}/sar_ascending.npy'
s1_descending = f'{base_path}/sar_descending.npy'

s1_acending_data = np.load(s1_acending)
s1_descending_data = np.load(s1_descending)

print(f'S1 Acending data shape: {s1_acending_data.shape}')
print(f'S1 Descending data shape: {s1_descending_data.shape}')


min_sar_time_step = min(s1_acending_data.shape[0], s1_descending_data.shape[0]) if s1_acending_data.shape[0] > 0 and s1_descending_data.shape[0] > 0 else s1_acending_data.shape[0] + s1_descending_data.shape[0]
import random
sar_time_step = random.randint(0, min_sar_time_step - 1)
print(f'Selected SAR time step: {sar_time_step}')

# 判断第一个维度是否为0，不为0即可可视化
plt.figure(figsize=(12, 6))
if s1_acending_data.shape[0] > 0:
    plt.subplot(1, 2, 1)
    plt.imshow(s1_acending_data[sar_time_step,:,:,0], cmap='gray')
    plt.title('S1 Acending')
    plt.axis('off')

if s1_descending_data.shape[0] > 0:
    plt.subplot(1, 2, 2)
    plt.imshow(s1_descending_data[sar_time_step,:,:,0], cmap='gray')
    plt.title('S1 Descending')
    plt.axis('off')

plt.tight_layout()
plt.savefig('/scratch/zf281/create_d-pixels_biomassters/s1_visualization.png', dpi=300)
plt.close()

### S2 data
s2_band = f'{base_path}/bands.npy'
s2_mask = f'{base_path}/masks.npy'

s2_band_data = np.load(s2_band) # （T，H, W, C）
s2_mask_data = np.load(s2_mask) # （T，H, W）
print(f'S2 Band data shape: {s2_band_data.shape}')
print(f'S2 Mask data shape: {s2_mask_data.shape}')
# 打印部分的mask值
print(f'S2 Mask data sample: {s2_mask_data[0, ::10, ::10]}')

# 对于mask的第二个和第三个维度求和，然后选出值最大的索引
mask_sum = np.sum(s2_mask_data, axis=(1, 2))  # 对每个时间步的mask求和
print(f'Mask sum: {mask_sum}')
max_mask_index = np.argmax(mask_sum)  # 找到最大值的索引
print(f'Selected S2 time step based on mask: {max_mask_index}')
# 可视化S2 Band数据
plt.figure(figsize=(6, 6))
s2_rgb = s2_band_data[max_mask_index, :, :, :3]  # 取三个波段
# 转为float
s2_rgb = s2_rgb.astype(np.float32)
# 归一化为0-1
for i in range(3):
    s2_rgb[:, :, i] = (s2_rgb[:, :, i] - np.min(s2_rgb[:, :, i])) / (np.max(s2_rgb[:, :, i]) - np.min(s2_rgb[:, :, i]))
plt.imshow(s2_rgb)
plt.title('S2 RGB Band')
plt.axis('off')
plt.tight_layout()
plt.savefig('/scratch/zf281/create_d-pixels_biomassters/s2_visualization.png', dpi=300)
plt.close()