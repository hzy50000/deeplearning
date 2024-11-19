import numpy as np
import matplotlib.pyplot as plt
import time

# folder = 'informer_2021_ftM_sl96_ll48_pl24_dm512_nh8_el2_dl1_df2048_atprob_fc5_ebtimeF_dtTrue_mxTrue_test_0'
pred = np.load(
    'pred.npy')
true = np.load(
    'true.npy')

print(pred.shape)
print(true.shape)

for i in range(10):
    plt.figure(1)
    plt.plot(true[i][0], label='GroundTruth')
    plt.plot(pred[i][0], label='Prediction')
    # plt.plot(true[i], label='GroundTruth')
    # plt.plot(pred[i], label='Prediction')
    time.sleep(0.5)
    # 我需要把(384, 24, 10)的数据改成(24, 384, 10)的形状 接下来是转换维度的代码：


    plt.legend()
    plt.show()