import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#将下面这条语句下的gdf文件所在路径改下
path=r'C:\Users\mcg\Desktop\class 2\4-class MI\BCICIV_2a_gdf\A03E.gdf'
raw=mne.io.read_raw_gdf(path)#读取数据，但没有导入

print(raw)

raw_data=raw.load_data()#加载原始数据，指的是纯粹的脑电数据，没有任何标签。
data=raw.get_data()#获得所有的脑电数据赋给data
print('显示脑电数据的shape：',data.shape)

info_des=raw.info#获得数据的具体描述，如通道，电极名称，采样率等
#info中有三个标签：也就是所以通道名称：cn_names,采样率：sfreq,

print('打印所有的通道名称 ：',raw_data.info['ch_names'])
print('显示采样频率 ：',raw_data.info['sfreq'])


montage = mne.channels.make_standard_montage('standard_1020')
montage.plot()

# 这个是数据库中的数据，用自己的就好了

data = mne.io.read_raw(path)
# 这边的设置是根据下面的报错结果来看哪些部分不符合规范，来进行手动甄别
# 修改通道名称需要传入一个字典，前面是通道的原名，后面是新名字
raw.rename_channels=(
    {'EEG-Fz': 'Fz', 'EEG-0': 'FC3', 'EEG-1': 'FC1', 'EEG-2': 'FCz', 'EEG-3': 'FC2', 'EEG-4': 'FC4', 'EEG-5': 'C5',
     'EEG-C3': 'C3', 'EEG-6': 'C1', 'EEG-Cz': 'Cz', 'EEG-7': 'C2', 'EEG-C4': 'C4', 'EEG-8': 'C6', 'EEG-9': 'CP3',
     'EEG-10': 'CP1', 'EEG-11': 'CPz', 'EEG-12': 'CP2', 'EEG-13': 'CP4', 'EEG-14': 'P1', 'EEG-Pz': 'Pz',
     'EEG-15': 'P2', 'EEG-16': 'POz', 'EOG-left': 'M1', 'EOG-central': 'Fpz', 'EOG-right': 'M2'})
data.rename_channels(raw.rename_channels)
# 这边设置on_missing = 'warn'，因为我不喜欢看报错
data.set_montage("standard_1020" , on_missing = 'warn')
data.plot()


#去除坏导
raw_cropped=raw.copy()
raw_cropped.crop(tmin=500,tmax=2000)
raw_cropped.plot(start=20,duration=1,n_channels=33,block=True,title='请检查并选中坏导')
raw_data.plot()

#重参考
raw_ref = raw_cropped.copy()
raw_ref.load_data()
raw_ref.set_eeg_reference(ref_channels=['M1','M2'])
raw_ref.plot(start=20,duration=1,block=True,title='重参考完成，无误请关闭窗口')
plt.show()

#滤波
raw_filter = raw_ref.copy()
raw_filter.filter(l_freq=1,h_freq=30)
raw_filter.notch_filter(freqs=50)
raw_filter.plot_psd(fmax=60)
plt.show(block=False)
raw_filter.plot(start=20,duration=1,block=True,title='滤波完成，准备ICA，无误请关闭窗口')

'''
# 独立成分分析
ica = mne.preprocessing.ICA(n_components=20, random_state=97, max_iter=800)
ica.fit(raw)
ica.exclude = [1, 2]  #   具体的细节有待完善
ica.plot_properties(raw, picks=ica.exclude)

orig_raw = raw.copy()
raw.load_data()
ica.apply(raw)'''

