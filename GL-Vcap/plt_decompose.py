from lib.Mydarw import *
from model.ASTGCN_r import *
import torch
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose

# 真实数据 （seq_l,n,f)
ele = "/Users/lvzhuanghu/PycharmProjects/D-GAT-LL-528/data/electricity.csv"
elec = np.loadtxt(ele, dtype = np.float32, delimiter = ",", skiprows = 0)
origi_data = np.load('/Users/lvzhuanghu/code/data/daily/8dayUnit/case_05_09_49_230_8day_sum.npy')
or_case = origi_data[:, :, 0]  # (seq_l,n,case
or_LST = origi_data[:, :, 1]
or_SR = origi_data[:, :, 2]
x_batch = origi_data[:, :, 1:3]
t,n,f = origi_data.shape
outputs_seq = []
for i in range(f):
    x_seq = origi_data[:, :, i]  # t,n
    out_seq_f = seasonal_decompose(x_seq, model='additive', period=30)
    outputs_seq.append(out_seq_f)
outputs_seq_ten = torch.stack(outputs_seq, 2)  # n f_out t


#pool分解
origi_data = torch.from_numpy(origi_data.transpose(0, 2, 1))
model = series_decomp(11)
res, trend = model(origi_data)
# 加分，乘法分解
result = seasonal_decompose(or_case, model='additive', period=30)
result_temp = seasonal_decompose(or_LST, model='additive', period=10)
result_rain = seasonal_decompose(or_SR, model='additive', period=10)
elctre = seasonal_decompose(elec, model='additive', period=10)
# result_mul = seasonal_decompose(or_case, model='multiplicative', period=10)
# result_mul_temp = seasonal_decompose(or_LST, model='multiplicative', period=10)
# result_mul_rain = seasonal_decompose(or_SR, model='multiplicative', period=10)

# 降雨数据的分解
# plt_2d(or_SR, res[:, :, 1], trend[:, :, 1], "or_rain", "season_rain", "Trend_rain", "Rain_pool_decm", 1)
# plt_2d(or_SR, result_rain.seasonal, result_rain.trend, "rain", "rain_season", "rain_Trend", "Rain_additive_decm", 1)
# plt_2d(or_SR, result_mul_rain.seasonal, result_mul_rain.trend, "rain", "rain_season", "rain_Trend",
#        "Rain_multiplicative_decm", 1)

# 温度数据的序列分解
# plt_2d(or_LST, res[:, :, 1], trend[:, :, 1], "or_Temp", "season_Temp", "Trend_Temp", "Temperature_pool_decm", 1)
# plt_2d(or_LST, result_temp.seasonal, result_temp.trend, "temperature", "temperature_season", "temperature_trend",
#        "Temperature_additive_decm", 1)
# plt_2d(or_LST, result_mul_temp.seasonal, result_mul_temp, "temperature", "temperature_season", "temperature_trend",
#        "Temperature_multiplicative_decm", 1)

# plt 病例数据的序列分解
# plt_2d(or_case, res[:, :, 2], trend[:, :, 2], "or_Rain", "season_Rain", "trend_Rain", "Case_pool_decm", 1)
# plt_2d(or_case, result.seasonal, result.trend, "case", "case_seasonal_", "case_trend", "Case_additive_decm", 1)
# plt_2d(result.resid, result.seasonal, result.trend, "case_resid", "case_seasonal", "case_trend", "Case_additive_decm_resi", 1)

plt_2d(elec, elctre.seasonal, elctre.trend, "electric_resid", "electric_seasonal", "electric_trend", "electric_additive_decm_resi", 1)
# plt_2d(or_case, result_mul.seasonal, result_mul.trend, "case", "case_seasonal_", "case_trend",
#        "Case_multiplicative_decm", 1)
