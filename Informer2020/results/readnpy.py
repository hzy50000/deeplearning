import numpy as np
import matplotlib.pyplot as plt

folder = 'informer_2021_ftM_sl96_ll48_pl24_dm512_nh8_el2_dl1_df2048_atprob_fc5_ebtimeF_dtTrue_mxTrue_test_0'
pred = np.load(
    '1/pred.npy')
true = np.load(
    '1/true.npy')

print(pred.shape)
print(true.shape)

plt.figure(1)
plt.plot(true[0, :, -1], label='GroundTruth')
plt.plot(pred[0, :, -1], label='Prediction')
plt.legend()
plt.show()