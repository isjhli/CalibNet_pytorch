import pykitti
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.pyplot import tight_layout

matplotlib.use('TkAgg')

basedir = "data"
seq = "00"

data = pykitti.odometry(basedir, seq)
print(len(data))

'''
calib: 具名元组
namedtupled 对象定义：collection.namedtuple(typename, field_names)
示例：
    import collections
    # User = collections.namedtuple('User', ['x', 'y', 'z'])
    User = collections.namedtuple('User', 'x y z')
    user = User('tester', '211', '523')
    ## 输出结果: User(x='tester', y='211', z='523')
    
    # 获取字段
    print(user._fields)
    
    # 获取数据
    print(user.x)
    print(user.y)
    print(user.z)
    
    
'''
calib = data.calib
intern = calib.K_cam0
extern = calib.T_cam0_velo

pcd = data.get_velo(150).T  # (4, N)
img = np.asarray(data.get_cam2(150))
H, W = img.shape[:2]

pcd[-1, :] = 1.0
pcd = extern @ pcd
pcd = intern @ pcd[:3, :]
u, v, w = pcd[0, :], pcd[1, :], pcd[2, :]
u = u / w
v = v / w
rev = (u >= 0) * (u < W) * (v >= 0) * (v < H) * (w > 0)
u = u[rev]
v = v[rev]
r = np.linalg.norm(pcd[:, rev], axis=0)
plt.figure(figsize=(12, 5), dpi=100, tight_layout=True)
plt.axis([0, W, H, 0])
plt.imshow(img)
plt.scatter([u], [v], c=[r], cmap='rainbow_r', alpha=0.5, s=2)
plt.savefig('demo_proj.png', bbox_inches='tight')
