
from features.features import FracDiff,Entropy,MicroStructucture
import numpy as np 
import polars as pl
import pandas as pd 
from finml.utils.stats import corrections 


def original_feature(df):
    label = df.select([pl.col('datetime'),pl.col("label")])
    factor = df.select([
            pl.col("close"),
            pl.col("volume"),
            pl.col('bids_value_level_0'),
            pl.col('bids_value_level_1'),
            pl.col('bids_value_level_2'),
            pl.col('bids_value_level_3'),
            pl.col('bids_value_level_4'),
            pl.col('asks_value_level_0'),
            pl.col('asks_value_level_1'),
            pl.col('asks_value_level_2'),
            pl.col('asks_value_level_3'),
            pl.col('asks_value_level_4'),
    ])
    corr = corrections(
        factor,
        label,
        corr_type ="pearson",
    )
    print(f"the  factor corrections are {corr.label}")

def ms_feature_evaluate(df):
    for i in range(50,500,50):
        ms = MicroStructucture(i,1)
        roll_effective_spread_0 = [] 
        roll_effective_spread_1 = [] 
        high_low_volatility = [] 
        alpha = [] 
        beta = [] 
        gamma = [] 
        corwin_schultz_volatility = [] 
        corwin_schultz_spread_0 = [] 
        corwin_schultz_spread_1 = [] 
        bar_based_kyle_lambda = [] 
        bar_based_amihud_lambda = [] 
        bar_based_hasbrouck_lambda = [] 
        trade_based_kyle_lambda_0 = [] 
        trade_based_amihud_lambda_0 = [] 
        trade_based_hasbrouck_lambda_0 = [] 
        trade_based_kyle_lambda_1 = [] 
        trade_based_amihud_lambda_1 = [] 
        trade_based_hasbrouck_lambda_1 = [] 
        vpin = [] 
        datetime = [] 
        for j in range(df.shape[0]):
            ms.update_raw(df[j,"high"],df[j,"low"],df[j,"close"],df[j,"volume"])
            datetime.append(df[j,"datetime"])
            if not ms.initialized():
                roll_effective_spread_0.append(np.nan)
                roll_effective_spread_1.append(np.nan)
                high_low_volatility.append(np.nan)
                alpha.append(np.nan)
                beta.append(np.nan)
                gamma.append(np.nan)
                corwin_schultz_volatility.append(np.nan)
                corwin_schultz_spread_0.append(np.nan)
                corwin_schultz_spread_1.append(np.nan)
                bar_based_kyle_lambda.append(np.nan)
                bar_based_amihud_lambda.append(np.nan)
                bar_based_hasbrouck_lambda.append(np.nan)
                trade_based_kyle_lambda_0.append(np.nan)
                trade_based_amihud_lambda_0.append(np.nan)
                trade_based_hasbrouck_lambda_0.append(np.nan)
                trade_based_kyle_lambda_1.append(np.nan)
                trade_based_amihud_lambda_1.append(np.nan)
                trade_based_hasbrouck_lambda_1.append(np.nan)
                vpin.append(np.nan)
            else:
                roll_effective_spread_0.append(ms.roll_effective_spread()[0])
                roll_effective_spread_1.append(ms.roll_effective_spread()[1])
                high_low_volatility.append(ms.high_low_volatility())
                beta.append(ms.get_beta())
                alpha.append(ms.get_alpha(beta[-1]))
                gamma.append(ms.get_gamma())
                corwin_schultz_volatility.append(ms.corwin_schultz_volatility(beta[-1]))
                corwin_schultz_spread_0.append(ms.corwin_schultz_spread()[0])
                corwin_schultz_spread_1.append(ms.corwin_schultz_spread()[1])
                bar_based_kyle_lambda.append(ms.bar_based_kyle_lambda())
                bar_based_amihud_lambda.append(ms.bar_based_amihud_lambda())
                bar_based_hasbrouck_lambda.append(ms.bar_based_hasbrouck_lambda())
                trade_based_kyle_lambda_0.append(ms.trades_based_kyle_lambda()[0])
                trade_based_amihud_lambda_0.append(ms.trades_based_amihud_lambda()[0])
                trade_based_hasbrouck_lambda_0.append(ms.trades_based_hasbrouck_lambda()[0])
                trade_based_kyle_lambda_1.append(ms.trades_based_kyle_lambda()[1])
                trade_based_amihud_lambda_1.append(ms.trades_based_amihud_lambda()[1])
                trade_based_hasbrouck_lambda_1.append(ms.trades_based_hasbrouck_lambda()[1])
                vpin.append(ms.vpin())
                
        factor = pd.DataFrame({
            "datetime":datetime,
            "roll_effective_spread_0" : roll_effective_spread_0,
            "roll_effective_spread_1" : roll_effective_spread_1,
            "high_low_volatility" : high_low_volatility,
            "alpha" : alpha,
            "beta" : beta,
            "gamma" : gamma,
            "corwin_schultz_volatility" : corwin_schultz_volatility,
            "corwin_schultz_spread_0" : corwin_schultz_spread_0,
            "corwin_schultz_spread_1" : corwin_schultz_spread_1,
            "bar_based_kyle_lambda" : bar_based_kyle_lambda,
            "bar_based_amihud_lambda" : bar_based_amihud_lambda,
            "bar_based_hasbrouck_lambda" : bar_based_hasbrouck_lambda,
            "trade_based_kyle_lambda_0":trade_based_kyle_lambda_0,
            "trade_based_amihud_lambda_0":trade_based_amihud_lambda_0,
            "trade_based_hasbrouck_lambda_0":trade_based_hasbrouck_lambda_0,
            "trade_based_kyle_lambda_1":trade_based_kyle_lambda_1,
            "trade_based_amihud_lambda_1":trade_based_amihud_lambda_1,
            "trade_based_hasbrouck_lambda_1":trade_based_hasbrouck_lambda_1,
            "vpin" : vpin
        })
        label = df.select([pl.col('datetime'),pl.col("label")])
        corr = corrections(
            factor,
            label,
            corr_type ="pearson",
        )
        print(f"the period {i} factor corrections are {corr.label}")


def test_entropy(df, labeled_df):
    for period in range(50,500,50):
        en = Entropy(period,10,-0.05,0.05)
        shannon_entropy = [] 
        plugin_entropy = [] 
        konto_entropy = [] 
        datetime = [] 
        for j in range(df.shape[0]):
            en.update_raw(df[j,"close"])
            datetime.append(df[j,"datetime"])
            if not en.initialized():
                shannon_entropy.append(np.nan)
                plugin_entropy.append(np.nan)
                konto_entropy.append(np.nan)
            else:
                shannon_entropy.append(en.shannon_entropy())
                plugin_entropy.append(en.plugin_entropy(period))
                konto_entropy.append(en.konto_entropy(0))
        factor = pd.DataFrame({
            "datetime":datetime,
            "shannon_entropy" : shannon_entropy,
            "plugin_entropy" : plugin_entropy,
            "konto_entropy" : konto_entropy,
        })
        label = labeled_df.select([pl.col('datetime'),pl.col("label")])
        corr = corrections(
            factor,
            label,
            corr_type ="pearson",
        )
        print(f"the period {period} factor corrections are {corr.label}")


def test_diff(df,col):
    label = df.select([pl.col('datetime'),pl.col("label")])
    for order in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
        for period in range(50,500,50):
            frac_dif = FracDiff(order,period)
            value = [] 
            datetime = [] 
            for j in range(df.shape[0]):
                    frac_dif.update_raw(df[j,col])
                    datetime.append(df[j,"datetime"])
                    if not frac_dif.initialized():
                        value.append(np.nan)
                    else:
                        value.append(frac_dif.value())
            factor = pd.DataFrame({
                    "datetime":datetime,
                    "value" : value,
                })
            corr = corrections(
                    factor,
                    label,
                    corr_type ="pearson",
                )
            print(f"the period {period} and order {order} factor corrections are {corr.label}")


def test_value_diff(df):
    label = df.select([pl.col('datetime'),pl.col("label")])
    for period in range(50,500,50):
        factor  = df.select([
            pl.col("datetime"),
            pl.col('bids_value_level_0').rolling_mean(period).alias("bids_value_level_0_mean"),
            pl.col('bids_value_level_1').rolling_mean(period).alias("bids_value_level_1_mean"),
            pl.col('bids_value_level_2').rolling_mean(period).alias("bids_value_level_2_mean"),
            pl.col('bids_value_level_3').rolling_mean(period).alias("bids_value_level_3_mean"),
            pl.col('bids_value_level_4').rolling_mean(period).alias("bids_value_level_4_mean"),
            pl.col('asks_value_level_0').rolling_mean(period).alias("asks_value_level_0_mean"),
            pl.col('asks_value_level_1').rolling_mean(period).alias("asks_value_level_1_mean"),
            pl.col('asks_value_level_2').rolling_mean(period).alias("asks_value_level_2_mean"),
            pl.col('asks_value_level_3').rolling_mean(period).alias("asks_value_level_3_mean"),
            pl.col('asks_value_level_4').rolling_mean(period).alias("asks_value_level_4_mean"),
            (pl.col('bids_value_level_0') - pl.col('asks_value_level_0')).alias("level_0_diff_mean").rolling_mean(period),
            (pl.col('bids_value_level_1') - pl.col('asks_value_level_1')).alias("level_1_diff_mean").rolling_mean(period),
            (pl.col('bids_value_level_2') - pl.col('asks_value_level_2')).alias("level_2_diff_mean").rolling_mean(period),
            (pl.col('bids_value_level_3') - pl.col('asks_value_level_3')).alias("level_3_diff_mean").rolling_mean(period),
            (pl.col('bids_value_level_4') - pl.col('asks_value_level_4')).alias("level_4_diff_mean").rolling_mean(period),
            (pl.col('bids_value_level_0') - pl.col('asks_value_level_0')+pl.col('bids_value_level_1') - pl.col('asks_value_level_1')+\
            pl.col('bids_value_level_2') - pl.col('asks_value_level_2')+pl.col('bids_value_level_3') - pl.col('asks_value_level_3')+\
            pl.col('bids_value_level_4') - pl.col('asks_value_level_4')).alias("total_diff_mean").rolling_mean(period),
            pl.col('bids_value_level_0').rolling_std(period).alias("bids_value_level_0_std"),
            pl.col('bids_value_level_1').rolling_std(period).alias("bids_value_level_1_std"),
            pl.col('bids_value_level_2').rolling_std(period).alias("bids_value_level_2_std"),
            pl.col('bids_value_level_3').rolling_std(period).alias("bids_value_level_3_std"),
            pl.col('bids_value_level_4').rolling_std(period).alias("bids_value_level_4_std"),
            pl.col('asks_value_level_0').rolling_std(period).alias("asks_value_level_0_std"),
            pl.col('asks_value_level_1').rolling_std(period).alias("asks_value_level_1_std"),
            pl.col('asks_value_level_2').rolling_std(period).alias("asks_value_level_2_std"),
            pl.col('asks_value_level_3').rolling_std(period).alias("asks_value_level_3_std"),
            pl.col('asks_value_level_4').rolling_std(period).alias("asks_value_level_4_std"),
            (pl.col('bids_value_level_0') - pl.col('asks_value_level_0')).alias("level_0_diff_std").rolling_std(period),
            (pl.col('bids_value_level_1') - pl.col('asks_value_level_1')).alias("level_1_diff_std").rolling_std(period),
            (pl.col('bids_value_level_2') - pl.col('asks_value_level_2')).alias("level_2_diff_std").rolling_std(period),
            (pl.col('bids_value_level_3') - pl.col('asks_value_level_3')).alias("level_3_diff_std").rolling_std(period),
            (pl.col('bids_value_level_4') - pl.col('asks_value_level_4')).alias("level_4_diff_std").rolling_std(period),
            (pl.col('bids_value_level_0') - pl.col('asks_value_level_0')+pl.col('bids_value_level_1') - pl.col('asks_value_level_1')+\
            pl.col('bids_value_level_2') - pl.col('asks_value_level_2')+pl.col('bids_value_level_3') - pl.col('asks_value_level_3')+\
            pl.col('bids_value_level_4') - pl.col('asks_value_level_4')).alias("total_diff_std").rolling_std(period),
            pl.col('bids_value_level_0').rolling_skew(period).alias("bids_value_level_0_skew"),
            pl.col('bids_value_level_1').rolling_skew(period).alias("bids_value_level_1_skew"),
            pl.col('bids_value_level_2').rolling_skew(period).alias("bids_value_level_2_skew"),
            pl.col('bids_value_level_3').rolling_skew(period).alias("bids_value_level_3_skew"),
            pl.col('bids_value_level_4').rolling_skew(period).alias("bids_value_level_4_skew"),
            pl.col('asks_value_level_0').rolling_skew(period).alias("asks_value_level_0_skew"),
            pl.col('asks_value_level_1').rolling_skew(period).alias("asks_value_level_1_skew"),
            pl.col('asks_value_level_2').rolling_skew(period).alias("asks_value_level_2_skew"),
            pl.col('asks_value_level_3').rolling_skew(period).alias("asks_value_level_3_skew"),
            pl.col('asks_value_level_4').rolling_skew(period).alias("asks_value_level_4_skew"),
            (pl.col('bids_value_level_0') - pl.col('asks_value_level_0')).alias("level_0_diff_skew").rolling_skew(period),
            (pl.col('bids_value_level_1') - pl.col('asks_value_level_1')).alias("level_1_diff_skew").rolling_skew(period),
            (pl.col('bids_value_level_2') - pl.col('asks_value_level_2')).alias("level_2_diff_skew").rolling_skew(period),
            (pl.col('bids_value_level_3') - pl.col('asks_value_level_3')).alias("level_3_diff_skew").rolling_skew(period),
            (pl.col('bids_value_level_4') - pl.col('asks_value_level_4')).alias("level_4_diff_skew").rolling_skew(period),
            (pl.col('bids_value_level_0') - pl.col('asks_value_level_0')+pl.col('bids_value_level_1') - pl.col('asks_value_level_1')+\
             pl.col('bids_value_level_2') - pl.col('asks_value_level_2')+pl.col('bids_value_level_3') - pl.col('asks_value_level_3')+\
             pl.col('bids_value_level_4') - pl.col('asks_value_level_4')).alias("total_diff_skew").rolling_skew(period),
            #((pl.col('bids_value_level_4')+pl.col('bids_value_level_3')+pl.col('bids_value_level_2'))/(pl.col('bids_value_level_0')+pl.col('bids_value_level_1')+pl.col('bids_value_level_2'))).alias("bids_high_low_rate_mean").rolling_mean(period),
            #((pl.col('asks_value_level_4')+pl.col('asks_value_level_3')+pl.col('asks_value_level_2'))/(pl.col('asks_value_level_0')+pl.col('asks_value_level_1')+pl.col('asks_value_level_2'))).alias("asks_high_low_rate_mean").rolling_mean(period),
        ])
        corr = corrections(
                    factor,
                    label,
                    corr_type ="pearson",
                )
        print(f"the period {period} factor corrections are {corr.label}")

from nautilus_trader.indicators.zscore import Zscore
def test_zscore(df):
    label = df.select([pl.col('datetime'),pl.col("label")])
    for period in range(50,500,50):
        bids_value_level_0_zscore = Zscore(period)
        bids_value_level_1_zscore = Zscore(period)
        bids_value_level_2_zscore = Zscore(period)
        bids_value_level_3_zscore = Zscore(period)
        bids_value_level_4_zscore = Zscore(period)
        asks_value_level_0_zscore = Zscore(period)
        asks_value_level_1_zscore = Zscore(period)
        asks_value_level_2_zscore = Zscore(period)
        asks_value_level_3_zscore = Zscore(period)
        asks_value_level_4_zscore = Zscore(period)
        value_level_0_diff_zscore = Zscore(period)
        value_level_1_diff_zscore = Zscore(period)
        value_level_2_diff_zscore = Zscore(period)
        value_level_3_diff_zscore = Zscore(period)
        value_level_4_diff_zscore = Zscore(period)
        datetime = [] 
        bids_value_level_0_zscore_list = [] 
        bids_value_level_1_zscore_list = []
        bids_value_level_2_zscore_list = []
        bids_value_level_3_zscore_list = []
        bids_value_level_4_zscore_list = []
        asks_value_level_0_zscore_list = []
        asks_value_level_1_zscore_list = []
        asks_value_level_2_zscore_list = []
        asks_value_level_3_zscore_list = []
        asks_value_level_4_zscore_list = []
        value_level_0_diff_zscore_list = []
        value_level_1_diff_zscore_list = []
        value_level_2_diff_zscore_list = []
        value_level_3_diff_zscore_list = []
        value_level_4_diff_zscore_list = []
        for j in range(df.shape[0]):
            bids_value_level_0_zscore.update_raw(df[j,"bids_value_level_0"])
            bids_value_level_1_zscore.update_raw(df[j,"bids_value_level_1"])
            bids_value_level_2_zscore.update_raw(df[j,"bids_value_level_2"])
            bids_value_level_3_zscore.update_raw(df[j,"bids_value_level_3"])
            bids_value_level_4_zscore.update_raw(df[j,"bids_value_level_4"])
            asks_value_level_0_zscore.update_raw(df[j,"asks_value_level_0"])
            asks_value_level_1_zscore.update_raw(df[j,"asks_value_level_1"])
            asks_value_level_2_zscore.update_raw(df[j,"asks_value_level_2"])
            asks_value_level_3_zscore.update_raw(df[j,"asks_value_level_3"])
            asks_value_level_4_zscore.update_raw(df[j,"asks_value_level_4"])
            value_level_0_diff_zscore.update_raw(df[j,"bids_value_level_0"]-df[j,"asks_value_level_0"])
            value_level_1_diff_zscore.update_raw(df[j,"bids_value_level_1"]-df[j,"asks_value_level_1"])
            value_level_2_diff_zscore.update_raw(df[j,"bids_value_level_2"]-df[j,"asks_value_level_2"])
            value_level_3_diff_zscore.update_raw(df[j,"bids_value_level_3"]-df[j,"asks_value_level_3"])
            value_level_4_diff_zscore.update_raw(df[j,"bids_value_level_4"]-df[j,"asks_value_level_4"])
            datetime.append(df[j,"datetime"])
            if not bids_value_level_0_zscore.initialized:
                bids_value_level_0_zscore_list.append(np.nan)
                bids_value_level_1_zscore_list.append(np.nan)
                bids_value_level_2_zscore_list.append(np.nan)
                bids_value_level_3_zscore_list.append(np.nan)
                bids_value_level_4_zscore_list.append(np.nan)
                asks_value_level_0_zscore_list.append(np.nan)
                asks_value_level_1_zscore_list.append(np.nan)
                asks_value_level_2_zscore_list.append(np.nan)
                asks_value_level_3_zscore_list.append(np.nan)
                asks_value_level_4_zscore_list.append(np.nan)
                value_level_0_diff_zscore_list.append(np.nan)
                value_level_1_diff_zscore_list.append(np.nan)
                value_level_2_diff_zscore_list.append(np.nan)
                value_level_3_diff_zscore_list.append(np.nan)
                value_level_4_diff_zscore_list.append(np.nan)
            else:
                bids_value_level_0_zscore_list.append(bids_value_level_0_zscore.value)
                bids_value_level_1_zscore_list.append(bids_value_level_1_zscore.value)
                bids_value_level_2_zscore_list.append(bids_value_level_2_zscore.value)
                bids_value_level_3_zscore_list.append(bids_value_level_3_zscore.value)
                bids_value_level_4_zscore_list.append(bids_value_level_4_zscore.value)
                asks_value_level_0_zscore_list.append(asks_value_level_0_zscore.value)
                asks_value_level_1_zscore_list.append(asks_value_level_1_zscore.value)
                asks_value_level_2_zscore_list.append(asks_value_level_2_zscore.value)
                asks_value_level_3_zscore_list.append(asks_value_level_3_zscore.value)
                asks_value_level_4_zscore_list.append(asks_value_level_4_zscore.value)
                value_level_0_diff_zscore_list.append(value_level_0_diff_zscore.value)
                value_level_1_diff_zscore_list.append(value_level_1_diff_zscore.value)
                value_level_2_diff_zscore_list.append(value_level_2_diff_zscore.value)
                value_level_3_diff_zscore_list.append(value_level_3_diff_zscore.value)
                value_level_4_diff_zscore_list.append(value_level_4_diff_zscore.value)
        factor = pd.DataFrame({
            "datetime":datetime,
            "bids_value_level_0_zscore" : bids_value_level_0_zscore_list,
            "bids_value_level_1_zscore" : bids_value_level_1_zscore_list,
            "bids_value_level_2_zscore" : bids_value_level_2_zscore_list,
            "bids_value_level_3_zscore" : bids_value_level_3_zscore_list,
            "bids_value_level_4_zscore" : bids_value_level_4_zscore_list,
            "asks_value_level_0_zscore" : asks_value_level_0_zscore_list,
            "asks_value_level_1_zscore" : asks_value_level_1_zscore_list,
            "asks_value_level_2_zscore" : asks_value_level_2_zscore_list,
            "asks_value_level_3_zscore" : asks_value_level_3_zscore_list,
            "asks_value_level_4_zscore" : asks_value_level_4_zscore_list,
            "value_level_0_diff_zscore" : value_level_0_diff_zscore_list,
            "value_level_1_diff_zscore" : value_level_1_diff_zscore_list,
            "value_level_2_diff_zscore" : value_level_2_diff_zscore_list,
            "value_level_3_diff_zscore" : value_level_3_diff_zscore_list,
            "value_level_4_diff_zscore" : value_level_4_diff_zscore_list,
        })
        label = df.select([pl.col('datetime'),pl.col("label")])
        corr = corrections(
            factor,
            label,
            corr_type ="pearson",
        )
        print(f"the period {period} factor corrections are {corr.label}")



from nautilus_trader.indicators.average.vidya import VariableIndexDynamicAverage
def test_vidya(df):
    label = df.select([pl.col('datetime'),pl.col("label")])
    for period in range(50,500,50):
        bids_value_level_0_vidya = VariableIndexDynamicAverage(period)
        bids_value_level_1_vidya = VariableIndexDynamicAverage(period)
        bids_value_level_2_vidya = VariableIndexDynamicAverage(period)
        bids_value_level_3_vidya = VariableIndexDynamicAverage(period)
        bids_value_level_4_vidya = VariableIndexDynamicAverage(period)
        asks_value_level_0_vidya = VariableIndexDynamicAverage(period)
        asks_value_level_1_vidya = VariableIndexDynamicAverage(period)
        asks_value_level_2_vidya = VariableIndexDynamicAverage(period)
        asks_value_level_3_vidya = VariableIndexDynamicAverage(period)
        asks_value_level_4_vidya = VariableIndexDynamicAverage(period)
        value_level_0_diff_vidya = VariableIndexDynamicAverage(period)
        value_level_1_diff_vidya = VariableIndexDynamicAverage(period)
        value_level_2_diff_vidya = VariableIndexDynamicAverage(period)
        value_level_3_diff_vidya = VariableIndexDynamicAverage(period)
        value_level_4_diff_vidya = VariableIndexDynamicAverage(period)
        datetime = [] 
        bids_value_level_0_vidya_list = [] 
        bids_value_level_1_vidya_list = []
        bids_value_level_2_vidya_list = []
        bids_value_level_3_vidya_list = []
        bids_value_level_4_vidya_list = []
        asks_value_level_0_vidya_list = []
        asks_value_level_1_vidya_list = []
        asks_value_level_2_vidya_list = []
        asks_value_level_3_vidya_list = []
        asks_value_level_4_vidya_list = []
        value_level_0_diff_vidya_list = []
        value_level_1_diff_vidya_list = []
        value_level_2_diff_vidya_list = []
        value_level_3_diff_vidya_list = []
        value_level_4_diff_vidya_list = []
        for j in range(df.shape[0]):
            bids_value_level_0_vidya.update_raw(df[j,"bids_value_level_0"])
            bids_value_level_1_vidya.update_raw(df[j,"bids_value_level_1"])
            bids_value_level_2_vidya.update_raw(df[j,"bids_value_level_2"])
            bids_value_level_3_vidya.update_raw(df[j,"bids_value_level_3"])
            bids_value_level_4_vidya.update_raw(df[j,"bids_value_level_4"])
            asks_value_level_0_vidya.update_raw(df[j,"asks_value_level_0"])
            asks_value_level_1_vidya.update_raw(df[j,"asks_value_level_1"])
            asks_value_level_2_vidya.update_raw(df[j,"asks_value_level_2"])
            asks_value_level_3_vidya.update_raw(df[j,"asks_value_level_3"])
            asks_value_level_4_vidya.update_raw(df[j,"asks_value_level_4"])
            value_level_0_diff_vidya.update_raw(df[j,"bids_value_level_0"]-df[j,"asks_value_level_0"])
            value_level_1_diff_vidya.update_raw(df[j,"bids_value_level_1"]-df[j,"asks_value_level_1"])
            value_level_2_diff_vidya.update_raw(df[j,"bids_value_level_2"]-df[j,"asks_value_level_2"])
            value_level_3_diff_vidya.update_raw(df[j,"bids_value_level_3"]-df[j,"asks_value_level_3"])
            value_level_4_diff_vidya.update_raw(df[j,"bids_value_level_4"]-df[j,"asks_value_level_4"])
            datetime.append(df[j,"datetime"])
            if not bids_value_level_0_vidya.initialized:
                bids_value_level_0_vidya_list.append(np.nan)
                bids_value_level_1_vidya_list.append(np.nan)
                bids_value_level_2_vidya_list.append(np.nan)
                bids_value_level_3_vidya_list.append(np.nan)
                bids_value_level_4_vidya_list.append(np.nan)
                asks_value_level_0_vidya_list.append(np.nan)
                asks_value_level_1_vidya_list.append(np.nan)
                asks_value_level_2_vidya_list.append(np.nan)
                asks_value_level_3_vidya_list.append(np.nan)
                asks_value_level_4_vidya_list.append(np.nan)
                value_level_0_diff_vidya_list.append(np.nan)
                value_level_1_diff_vidya_list.append(np.nan)
                value_level_2_diff_vidya_list.append(np.nan)
                value_level_3_diff_vidya_list.append(np.nan)
                value_level_4_diff_vidya_list.append(np.nan)
            else:
                bids_value_level_0_vidya_list.append(bids_value_level_0_vidya.value)
                bids_value_level_1_vidya_list.append(bids_value_level_1_vidya.value)
                bids_value_level_2_vidya_list.append(bids_value_level_2_vidya.value)
                bids_value_level_3_vidya_list.append(bids_value_level_3_vidya.value)
                bids_value_level_4_vidya_list.append(bids_value_level_4_vidya.value)
                asks_value_level_0_vidya_list.append(asks_value_level_0_vidya.value)
                asks_value_level_1_vidya_list.append(asks_value_level_1_vidya.value)
                asks_value_level_2_vidya_list.append(asks_value_level_2_vidya.value)
                asks_value_level_3_vidya_list.append(asks_value_level_3_vidya.value)
                asks_value_level_4_vidya_list.append(asks_value_level_4_vidya.value)
                value_level_0_diff_vidya_list.append(value_level_0_diff_vidya.value)
                value_level_1_diff_vidya_list.append(value_level_1_diff_vidya.value)
                value_level_2_diff_vidya_list.append(value_level_2_diff_vidya.value)
                value_level_3_diff_vidya_list.append(value_level_3_diff_vidya.value)
                value_level_4_diff_vidya_list.append(value_level_4_diff_vidya.value)
        factor = pd.DataFrame({
            "datetime":datetime,
            "bids_value_level_0_vidya" : bids_value_level_0_vidya_list,
            "bids_value_level_1_vidya" : bids_value_level_1_vidya_list,
            "bids_value_level_2_vidya" : bids_value_level_2_vidya_list,
            "bids_value_level_3_vidya" : bids_value_level_3_vidya_list,
            "bids_value_level_4_vidya" : bids_value_level_4_vidya_list,
            "asks_value_level_0_vidya" : asks_value_level_0_vidya_list,
            "asks_value_level_1_vidya" : asks_value_level_1_vidya_list,
            "asks_value_level_2_vidya" : asks_value_level_2_vidya_list,
            "asks_value_level_3_vidya" : asks_value_level_3_vidya_list,
            "asks_value_level_4_vidya" : asks_value_level_4_vidya_list,
            "value_level_0_diff_vidya" : value_level_0_diff_vidya_list,
            "value_level_1_diff_vidya" : value_level_1_diff_vidya_list,
            "value_level_2_diff_vidya" : value_level_2_diff_vidya_list,
            "value_level_3_diff_vidya" : value_level_3_diff_vidya_list,
            "value_level_4_diff_vidya" : value_level_4_diff_vidya_list,
        })
        label = df.select([pl.col('datetime'),pl.col("label")])
        corr = corrections(
            factor,
            label,
            corr_type ="pearson",
        )
        print(f"the period {period} factor corrections are {corr.label}")