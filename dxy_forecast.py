import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import statsmodels.api as sm
from pandas.tseries.offsets import QuarterEnd, QuarterBegin, MonthEnd, MonthBegin, BDay
import math
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet as ENet
from sklearn.inspection import PartialDependenceDisplay

from sklearn.ensemble import RandomForestRegressor as RF
from sklearn.ensemble import ExtraTreesRegressor as EXT

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LeakyReLU, Dropout
from tensorflow.keras import regularizers
from tensorflow.keras import backend

import statsmodels.formula.api as smf

from time import time

from tqdm import tqdm

import itertools

idx = pd.IndexSlice


######################################### country lists ##########################################
clist2 = ['DE', 'IT', 'FR', 'NL']
oths = ['GB', 'CH', 'JP', 'CA', 'AU', 'NZ', 'SE', 'NO', 'DK']
co14 = ['GB', 'CH', 'JP', 'CA', 'AU', 'NZ', 'SE', 'NO', 'DK', 'EU', 'DE', 'IT', 'FR', 'NL']
co13 = ['GB', 'CH', 'JP', 'CA', 'AU', 'NZ', 'SE', 'NO', 'DK', 'DE', 'IT', 'FR', 'NL']
co17 = ['GB', 'CH', 'JP', 'CA', 'AU', 'NZ', 'SE', 'NO', 'DK', 'EU', 'DE', 'IT', 'FR', 'NL'] + ['US', 'KR', 'CN']
co10 = ['GB', 'CH', 'JP', 'CA', 'AU', 'NZ', 'SE', 'NO', 'DK', 'EU']
co11 = ['GB', 'CH', 'JP', 'CA', 'AU', 'NZ', 'SE', 'NO', 'DK', 'EU', 'US']
co12 = ['GB', 'CH', 'JP', 'CA', 'AU', 'NZ', 'SE', 'NO', 'DK', 'EU', 'US', 'KR']
co6  = ['GB', 'CH', 'JP', 'CA', 'SE', 'EU']

oecd_iso3 = ['AUT', 'BEL', 'CAN', 'CZE', 'DNK', 'FIN', 'FRA', 'DEU', 'GRC',
             'HUN', 'ISL', 'IRL', 'ITA', 'JPN', 'KOR', 'LUX', 'MEX', 'NLD',
             'NOR', 'POL', 'PRT', 'SVK', 'ESP', 'SWE', 'CHE', 'TUR', 'GBR',
             'USA', 'BRA', 'CHL', 'CHN', 'COL', 'CRI', 'EST', 'IND', 'IDN',
             'ISR', 'LVA', 'LTU', 'RUS', 'SAU', 'SVN', 'ZAF', 'OECD', 'OECDE',
             'G-7', 'G-20', 'EA19', 'EU27_2020', 'AUS', 'NZL']

oecd_iso2 = ['AUT', 'BEL', 'CA', 'CZE', 'DK', 'FIN', 'FR', 'DE', 'GRC',
             'HUN', 'ISL', 'IRL', 'IT', 'JP', 'KR', 'LUX', 'MEX', 'NL',
             'NO', 'POL', 'PRT', 'SVK', 'ESP', 'SE', 'CH', 'TUR', 'GB',
             'US', 'BRA', 'CHL', 'CN', 'COL', 'CRI', 'EST', 'IND', 'IDN',
             'ISR', 'LVA', 'LTU', 'RUS', 'SAU', 'SVN', 'ZAF', 'OECD', 'OECDE',
             'G-7', 'G-20', 'EU', 'EU27_2020', 'AU', 'NZ']


dxy_weight = pd.Series([0.119, 0.036, 0.136, 0.091, 0.042, 0.576], index=co6)

######################################### functions for data and filter ##########################################
def get_breit_excel3(req_ids, alias_nm, 
                     start_d=None, end_d=None,
                     period_trim=False):
    # API 호출
    API = "http://datahub.boknet.intra/api/v1/obs/lists"
    res = requests.post(API, data={"ids":req_ids})
    data_list = res.json()["data"][0]
    
    # API 호출로 받은 결과를 Data Frame으로 저장
    data = pd.DataFrame()
    for alias, value in zip(alias_nm, data_list):
        try:
            df = pd.DataFrame(value["observations"])
            df.set_index("period", inplace=True)
            df.index = pd.to_datetime(df.index)
            df.columns = [alias]
            data = df.copy() if not len(data) else data.join(df, how="outer")
        except:
            print(f'{alias} is not imported.')
    
    # 옵션에 따라 시작일, 종료일, Trim 적용
    if start_d:
        data = data[data.index >= start_d]
    if end_d:
        data = data[data.index <= end_d]
    if period_trim:
        data.index = data.index.to_period('M')
    return data


def get_fx_rates(contries, sm='1999', em='2022-9'):
    df = pd.read_excel('data/Exchange Rates.xlsx', sheet_name='SPOT')

    b = df.iloc[:, 1:42:3]
    b.columns = co14

    s = df.iloc[:, 2:42:3]
    s.columns = co14

    S = pd.concat([b, s], axis=1, keys=['B', 'S'])
    S.index = df['Name']

    # USD to GBP => GBP to USD
    S.loc[:, idx['B', 'GB']], S.loc[:, idx['S', 'GB']] = 1 / S.loc[:, idx['S', 'GB']], 1 / S.loc[:, idx['B', 'GB']]

    S.index.names = ['Date']

    ch = pd.read_excel('data/Exchange Rates CNY.xlsx', sheet_name='SPOT', skiprows=4)
    ch.columns = ['Date', 'S', 'B']
    ch = ch.set_index('Date')
    ch.loc[ch.S.le(0), 'S'] = np.nan
    ch.loc[ch.B.le(0), 'B'] = np.nan

    kr = pd.read_excel('data/Exchange Rates KRW.xlsx', sheet_name='SPOT', skiprows=4)
    kr.columns = ['Date', 'S', 'B']
    kr = kr.set_index('Date')
    kr.loc[kr.S.le(0), 'S'] = np.nan
    kr.loc[kr.B.le(0), 'B'] = np.nan

    ch = pd.concat([ch], axis=1, keys=['CN']).reorder_levels([1, 0], axis=1)
    kr = pd.concat([kr], axis=1, keys=['KR']).reorder_levels([1, 0], axis=1)

    S = pd.concat([S, ch, kr], axis=1).sort_index(axis=1)
    Sm = (S['B'] + S['S'])/2
    S = pd.concat([S, pd.concat([Sm], axis=1, keys=['M'])], axis=1)

    df = pd.read_excel('data/Exchange Rates.xlsx', sheet_name='FORWARD')

    b = df.iloc[:, 1:42:3]
    b.columns = co14

    s = df.iloc[:, 2:42:3]
    s.columns = co14

    F = pd.concat([b, s], axis=1, keys=['B', 'S'])
    F.index = df['Name']

    # USD to GBP => GBP to USD
    F.loc[:, idx['B', 'GB']], F.loc[:, idx['S', 'GB']] = 1 / F.loc[:, idx['S', 'GB']], 1 / F.loc[:, idx['B', 'GB']]

    F.index.names = ['Date']

    ch = pd.read_excel('data/Exchange Rates CNY.xlsx', sheet_name='FORWARD', skiprows=4)
    ch.columns = ['Date', 'S', 'B']
    ch = ch.set_index('Date')
    ch.loc[ch.S.le(0), 'S'] = np.nan
    ch.loc[ch.B.le(0), 'B'] = np.nan

    kr = pd.read_excel('data/Exchange Rates KRW.xlsx', sheet_name='FORWARD', skiprows=4)
    kr.columns = ['Date', 'S', 'B']
    kr = kr.set_index('Date')
    kr.loc[kr.S.le(0), 'S'] = np.nan
    kr.loc[kr.B.le(0), 'B'] = np.nan

    ch = pd.concat([ch], axis=1, keys=['CN']).reorder_levels([1, 0], axis=1)
    kr = pd.concat([kr], axis=1, keys=['KR']).reorder_levels([1, 0], axis=1)

    F = pd.concat([F, ch, kr], axis=1).sort_index(axis=1)
    Fm = (F['B'] + F['S'])/2
    F = pd.concat([F, pd.concat([Fm], axis=1, keys=['M'])], axis=1)
    
    S = S.loc[sm:em, idx[:, contries]].dropna(how='all', axis=0).sort_index(axis=1)
    F = F.loc[sm:em, idx[:, contries]].dropna(how='all', axis=0).sort_index(axis=1)
    
    return S, F


def filippou_table1(dS, sm, em, clist):
    
    s3 = pd.concat([dS.loc[sm:em, clist].mean() * 12], axis=0).to_frame('ann. mean %') * 100
    s4 = pd.concat([dS.loc[sm:em, clist].std() * np.sqrt(12)], axis=0).to_frame('ann. vol %') * 100
    s5 = pd.concat([dS.loc[sm:em, clist].skew()], axis=0).to_frame('skewness')
    s6 = pd.concat([dS.loc[sm:em, clist].kurtosis()], axis=0).to_frame('excess kurtosis')
    s7 = pd.concat([dS.loc[sm:em, clist].apply(lambda x: x.autocorr(), axis=0)], axis=0).to_frame('autocorr')
    
    return pd.concat([s3, s4, s5, s6, s7], axis=1)


def bp_cycle(df, p0, p1, normalization=True):
    '''Returns: cycle, which is normalized by default'''
    cycles = df.copy()
    for co in df.columns:
        cycle, trend = sm.tsa.filters.cffilter(df.loc[:, co].dropna(), low = p0, high = p1, drift=True)
        if normalization == True:
            cycle = cycle / cycle.std()
        cycles.loc[:, co] = cycle
    return cycles


def os_bp_filter(df, p0, p1, drift=False):
    '''Returns: cycle, which is normalized by default'''
    
    trend = df.copy()
    cycle = df.copy()
    for co in df.columns:
        for t in df.index:
            try:
                cy, tr = sm.tsa.filters.cffilter(df.loc[:t, co].dropna(), low = p0, high = p1, drift=drift)
                trend.loc[t, co] = tr.loc[t]
                cycle.loc[t, co] = cy.loc[t]
            except:
                continue
                
    return cycle, trend


################################################################################
#################            NOT USED         ##################################
################################################################################
# unem = pd.read_excel('data/Unemployment Rate_OECD.xlsx')
# unem = unem.replace(oecd_iso3, oecd_iso2)
# unem = unem.loc[unem.LOCATION.isin(co12)]

# unem = unem.loc[unem.SUBJECT.eq('TOT')]
# unem = unem.drop(['INDICATOR', 'SUBJECT', 'MEASURE', 'Flag Codes'], axis=1)

# nz = unem.loc[unem.LOCATION.isin(['NZ']) & unem.FREQUENCY.eq('Q')]
# nz = nz.drop('FREQUENCY', axis=1)
# nz = nz.set_index('TIME')
# nz = nz.set_index('LOCATION', append=True).unstack()['Value']
# nz.index = pd.to_datetime(nz.index) + QuarterEnd()
# nz = nz.resample('M').ffill()

# # ch = pd.read_excel('data/snb-data-amarbma-en-selection-20221021_0900.xlsx', skiprows=20, index_col=0)
# # ch = ch.iloc[:, :-1]
# # ch.columns = ['CH']
# # ch.index = pd.to_datetime(ch.index) + MonthEnd()

# unem = unem.loc[unem.FREQUENCY.eq('M')]
# unem = unem.drop('FREQUENCY', axis=1)
# unem = unem.set_index('TIME')
# unem = unem.set_index('LOCATION', append=True).unstack()['Value']
# unem.index = unem.index + MonthEnd()
# unem = pd.concat([unem, nz], axis=1).loc[b_date:e_date]


# cpi = pd.read_excel('data/CPI.xlsx')
# index = cpi['Name']
# cpi = cpi.iloc[:, 1:32:2]
# reuters = ['FR CPI (2015=100) NADJ', 'NL CPI NADJ', 'IT CPI INCLUDING TOBACCO - NIC (LINKED & REBASED) NADJ',
#            'SW CPI (2020M12=100) NADJ', 'CH CPI (CPPY=100) NADJ', 'UK CPI INDEX 00 : ALL ITEMS- ESTIMATED PRE-1997 NADJ',
#            'US CPI - ALL URBAN SAMPLE: ALL ITEMS NADJ', 'BD CPI: TOTAL NADJ', 'JP CPI: NATIONAL MEASURE NADJ', 
#            'CN CPI NADJ', 'KO CPI NADJ', 'NW CPI NADJ', 'DK CPI NADJ', 'SD CPI NADJ', 'AU CPI NADJ', 'NZ CPI NADJ']
# reu_iso2 = ['FR', 'NL', 'IT', 'CH', 'CN', 'UK', 'US', 'DE', 'JP', 'CA', 'KR', 'NO', 'DK', 'SE', 'AU', 'NZ']
# cpi = cpi.rename(columns=dict(zip(reuters, reu_iso2)))
# cpi.index = index + MonthEnd()

# id = ["NECOS-902U008-M-KR", "NECOS-902U008-M-CA", "NECOS-902U008-M-CN", "NECOS-902U008-M-DK",
#       "NECOS-902U008-M-FR", "NECOS-902U008-M-DE", "NECOS-902U008-M-IT", "NECOS-902U008-M-JP", "NECOS-902U008-M-NL",
#       "NECOS-902U008-M-NO", "NECOS-902U008-M-SE", "NECOS-902U008-M-CH", "NECOS-902U008-M-GB", "NECOS-902U008-M-US"]
# nm = ["KR", "CA", "CN", "DK", "FR", "DE", "IT", "JP", "NL", "NO", "SE", "CH", "GB", "US"]
# cpi_breit = get_breit_excel3(id, nm)
# cpi_breit.index = cpi_breit.index + MonthEnd()
# cpi_breit = cpi_breit.astype(float)

# cpi_aunz = get_breit_excel3(["NECOS-902U008-Q-AU", "NECOS-902U008-Q-NZ"], ['AU', 'NZ'])
# cpi_aunz.index = cpi_aunz.index + QuarterEnd()
# cpi_aunz = cpi_aunz.resample('M').bfill()
# cpi_aunz = cpi_aunz.astype(float)

# cpi_breit = pd.concat([cpi_breit, cpi_aunz], axis=1)
# cpi[['AU', 'NZ']] = cpi_aunz[['AU', 'NZ']]



##############   FITTING   #################

def read_Xy(X, y):
    """
    X : X.pkl, Xsd.pkl, Xsdm.pkl
    y : y.pkl, ym.pkl
    """
    X = pd.read_pickle(X)
    y = pd.read_pickle(y)
    X = X.stack().dropna()
    y = y.stack().reindex(X.index)
    
    return X, y


def ERIC_enet(X, y, enet):
    
    lam = enet.alpha
    err = y - enet.predict(X)
    
    ssr = np.sum(err**2)
    
    # effective degree of freedom
    lam2 = lam*(1 - enet.l1_ratio)*0.5
    A = X.columns[enet.coef_ != 0]
    Xa = X[A].values
    
    NT = Xa.shape[0]
    J = Xa.shape[1]
    
    Ha = Xa.dot(np.linalg.inv((Xa.T.dot(Xa) + lam2*np.eye(J)))).dot(Xa.T)
    df = np.trace(Ha)

    eric = NT*np.log(ssr/NT) + df*np.log(ssr/lam)
    
    return eric, df


def R_squared(X, y, enet):
    
    err = y - enet.predict(X)
    
    return (1 - np.sum(err**2)/np.sum(y**2))


def build_dnn(l1_reg, l2_reg, n_input):
    dnn = Sequential()
    dnn.add(Dense(16, activation=LeakyReLU(alpha=0.01), use_bias=False, input_shape=(n_input,),    
                  kernel_regularizer=regularizers.L1L2(l1=l1_reg, l2=l2_reg) ))
    dnn.add(Dropout(0.5))
    dnn.add(Dense(8, activation=LeakyReLU(alpha=0.01), use_bias=False,
                  kernel_regularizer=regularizers.L1L2(l1=l1_reg, l2=l2_reg) ))
    dnn.add(Dropout(0.5))
    dnn.add(Dense(4, activation=LeakyReLU(alpha=0.01), use_bias=False,
                  kernel_regularizer=regularizers.L1L2(l1=l1_reg, l2=l2_reg) ))
    dnn.add(Dropout(0.5))
    dnn.add(Dense(2, activation=LeakyReLU(alpha=0.01), use_bias=False,
                  kernel_regularizer=regularizers.L1L2(l1=l1_reg, l2=l2_reg) ))
    dnn.add(Dense(1, activation=LeakyReLU(alpha=0.01), use_bias=False,
                  kernel_regularizer=regularizers.L1L2(l1=l1_reg, l2=l2_reg) ))
    dnn.compile(optimizer='adam', loss='mse', metrics=['mse'])  
    return dnn


#########################################################################################################
def highlight_cells(val):
    color = 'yellow' if val < 0.01 else ('lightyellow' if val < 0.05 else '')
    return f'background-color: {color}'