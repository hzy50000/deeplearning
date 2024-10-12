import numpy as np
import matplotlib.pyplot as plt

folder = 'informer_ETTh1_ftM_sl96_ll48_pl23_dm512_nh8_el2_dl1_df2048_atprob_fc5_ebtimeF_dtTrue_mxTrue_test_0'
pred = np.load(
    'informer_2021_ftMS_sl96_ll48_pl24_dm512_nh8_el2_dl1_df2048_atprob_fc5_ebtimeF_dtTrue_mxTrue_test_1/pred.npy')
true = np.load(
    'informer_2021_ftMS_sl96_ll48_pl24_dm512_nh8_el2_dl1_df2048_atprob_fc5_ebtimeF_dtTrue_mxTrue_test_1/true.npy')

print(pred.shape)
print(true.shape)

plt.figure(1)
plt.plot(true[0, :, -1], label='GroundTruth')
plt.plot(pred[0, :, -1], label='Prediction')
plt.legend()
plt.show()