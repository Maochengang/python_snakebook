import numpy as np
import mne
#载入数据

sample_data_folder = mne.datasets.sample.data_path()
sample_data_raw_file = (sample_data_folder / 'MEG' / 'sample' /
                        'sample_audvis_filt-0-40_raw.fif')
raw = mne.io.read_raw_fif(sample_data_raw_file)
print(raw)
print(raw.info)

raw.plot_psd(fmax=50)#plot psd显示每种传感器类型的功率谱密度（Power Spectral Density，PSD）如 fmax=50表示仅绘制50 Hz以下的频率
raw.plot(duration=5, n_channels=30)#绘制的原始传感器轨迹图  duration_持续时间     channel_通道数

# 独立成分分析
ica = mne.preprocessing.ICA(n_components=20, random_state=97, max_iter=800)
ica.fit(raw)
ica.exclude = [1, 2]  #   具体的细节有待完善
ica.plot_properties(raw, picks=ica.exclude)

orig_raw = raw.copy()
raw.load_data()
ica.apply(raw)
'''
# 独立分析前后对比
chs = ['MEG 0111', 'MEG 0121', 'MEG 0131', 'MEG 0211', 'MEG 0221', 'MEG 0231',
       'MEG 0311', 'MEG 0321', 'MEG 0331', 'MEG 1511', 'MEG 1521', 'MEG 1531',
       'EEG 001', 'EEG 002', 'EEG 003', 'EEG 004', 'EEG 005', 'EEG 006',
       'EEG 007', 'EEG 008']
chan_idxs = [raw.ch_names.index(ch) for ch in chs]
orig_raw.plot(order=chan_idxs, start=12, duration=4)
raw.plot(order=chan_idxs, start=12, duration=4)

# 检测实验事件，给事件命名
events = mne.find_events(raw, stim_channel='STI 014')
events = mne.find_events(raw, stim_channel='STI 014')
print(events[:5])  # show the first 5    #生成的事件数组是一个普通的 3 列 NumPy 数组，第一列是样本号，最后一列是整数事件 ID；中间列通常被忽略。我们可以提供一个事件字典，将整数 ID 映射到实验条件或事件，而不是跟踪整数事件 ID。在这个数据集中，
#映射如下所示：Event ID
event_dict = {'auditory/left': 1, 'auditory/right': 2, 'visual/left': 3,
              'visual/right': 4, 'smiley': 5, 'buttonpress': 32}
fig = mne.viz.plot_events(events, event_id=event_dict, sfreq=raw.info['sfreq'],
                          first_samp=raw.first_samp)



#去除坏段
reject_criteria = dict(mag=4000e-15,     # 4000 fT
                       grad=4000e-13,    # 4000 fT/cm
                       eeg=150e-6,       # 150 µV
                       eog=250e-6)       # 250 µV
epochs = mne.Epochs(raw, events, event_id=event_dict, tmin=-0.2, tmax=0.5,
                    reject=reject_criteria, preload=True)'''

#滤波
raw_filter = raw_ref.copy()
raw_filter.filter(l_freq=1,h_freq=30)
raw_filter.notch_filter(freqs=50)
raw_filter.plot_psd(fmax=60)
plt.show(block=False)
raw_filter.plot(start=20,duration=1,block=True,title='滤波完成，准备ICA，无误请关闭窗口')
