
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
    print(f"the  factor corrections are {corr}")

def ms_feature_evaluate(
    df,
    columns=["high","low","close","volume"],
    start_period = 5,
    end_period = 100,
    step = 50,
):
    results = []
    for period in np.arange(start_period,end_period,step):
        ms = MicroStructucture(period,1)
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
            ms.update_raw(df[j,columns[0]],df[j,columns[1]],df[j,columns[2]],df[j,columns[3]])
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
            f"{period}_roll_effective_spread_0" : roll_effective_spread_0,
            f"{period}_roll_effective_spread_1" : roll_effective_spread_1,
            f"{period}_high_low_volatility" : high_low_volatility,
            f"{period}_alpha" : alpha,
            f"{period}_beta" : beta,
            f"{period}_gamma" : gamma,
            f"{period}_corwin_schultz_volatility" : corwin_schultz_volatility,
            f"{period}_corwin_schultz_spread_0" : corwin_schultz_spread_0,
            f"{period}_corwin_schultz_spread_1" : corwin_schultz_spread_1,
            f"{period}_bar_based_kyle_lambda" : bar_based_kyle_lambda,
            f"{period}_bar_based_amihud_lambda" : bar_based_amihud_lambda,
            f"{period}_bar_based_hasbrouck_lambda" : bar_based_hasbrouck_lambda,
            f"{period}_trade_based_kyle_lambda_0":trade_based_kyle_lambda_0,
            f"{period}_trade_based_amihud_lambda_0":trade_based_amihud_lambda_0,
            f"{period}_trade_based_hasbrouck_lambda_0":trade_based_hasbrouck_lambda_0,
            f"{period}_trade_based_kyle_lambda_1":trade_based_kyle_lambda_1,
            f"{period}_trade_based_amihud_lambda_1":trade_based_amihud_lambda_1,
            f"{period}_trade_based_hasbrouck_lambda_1":trade_based_hasbrouck_lambda_1,
            f"{period}_vpin" : vpin
        })
        label = df.select([pl.col('datetime'),pl.col("label")])
        corr = corrections(
            factor,
            label,
            corr_type ="spearman",
        )
        print(f"the period {period} factor corrections are {corr}")
        results.append(corr)
    return pd.concat(results,axis=0)


def test_entropy(
    df,
    col,
    start_period,
    end_period,
    step,
):
    label = df.select([pl.col('datetime'),pl.col("label")])
    results = []
    for period in range(start_period,end_period,step):
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
            f"{period}shannon_entropy" : shannon_entropy,
            f"{period}plugin_entropy" : plugin_entropy,
            f"{period}konto_entropy" : konto_entropy,
        })
        corr = corrections(
            factor,
            label,
            corr_type ="spearman",
        )
        print(f"the period {period} factor corrections are {corr}")
        results.append(corr)
    return pd.concat(results,axis=0)


def test_diff(
    df,
    col
):
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
            print(f"the period {period} and order {order} factor corrections are {corr}")


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
        print(f"the period {period} factor corrections are {corr}")

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
        print(f"the period {period} factor corrections are {corr}")



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
        print(f"the period {period} factor corrections are {corr}")


from nautilus_trader.indicators.zscore import Zscore
def zscore_analysis(
    _df,
    col,
    start_period = 6,
    end_period = 300,
    step = 30,
):
    results = []
    label = _df.select([pl.col('datetime'),pl.col("label")])
    for period in range(start_period,end_period,step):
        zscore = Zscore(period)
        datetime = [] 
        zscore_list = [] 
    
        for j in range(_df.shape[0]):
            zscore.update_raw(_df[j,col])
            datetime.append(_df[j,"datetime"])
            if not zscore.initialized:
                zscore_list.append(np.nan)
            else:
                zscore_list.append(zscore.value)
        factor = pd.DataFrame({
            "datetime":datetime,
            f"zscore_{period}" : zscore_list,
        })
        label = _df.select([pl.col('datetime'),pl.col("label")])
        corr = corrections(
            factor,
            label,
            corr_type ="spearman",
        )
        print(f"the period {period} factor corrections are {corr}")
        results.append(corr) 
    return pd.concat(results,axis=0)


from nautilus_trader.indicators.rvi import RelativeVolatilityIndex
def test_rvi(df,col):
    label = df.select([pl.col('datetime'),pl.col("label")])
    for period in range(50,500,50):
        rvi = RelativeVolatilityIndex(period)
        datetime = [] 
        rvi_list = [] 
    
        for j in range(df.shape[0]):
            rvi.update_raw(df[j,col])
            datetime.append(df[j,"datetime"])
            if not rvi.initialized:
                rvi_list.append(np.nan)
            else:
                rvi_list.append(rvi.value)
        factor = pd.DataFrame({
            "datetime":datetime,
            "rvi" : rvi_list,
        })
        label = df.select([pl.col('datetime'),pl.col("label")])
        corr = corrections(
            factor,
            label,
            corr_type ="pearson",
        )
        print(f"the period {period} factor corrections are {corr}")


from nautilus_trader.indicators.pgo import PrettyGoodOscillator 
def test_pgo(df):
    label = df.select([pl.col('datetime'),pl.col("label")])
    for period in range(50,500,50):
        pgo =PrettyGoodOscillator(period)
        datetime = [] 
        pgo_list = [] 
    
        for j in range(df.shape[0]):
            pgo.update_raw(df[j,"high"],df[j,"low"],df[j,"close"])
            datetime.append(df[j,"datetime"])
            if not pgo.initialized:
                pgo_list.append(np.nan)
            else:
                pgo_list.append(pgo.value)
        factor = pd.DataFrame({
            "datetime":datetime,
            "pgo" : pgo_list,
        })
        label = df.select([pl.col('datetime'),pl.col("label")])
        corr = corrections(
            factor,
            label,
            corr_type ="pearson",
        )
        print(f"the period {period} factor corrections are {corr}")

from features.features import RollStats
def test_value_diff(df):
    label = df.select([pl.col('datetime'),pl.col("label")])
    for period in range(10,100,10):
        factor  = df.select([
            pl.col("datetime"),
            pl.col('small_buy_value').rolling_mean(period).alias("small_buy_value_mean"),
            pl.col('big_buy_value').rolling_mean(period).alias("big_buy_value_mean"),
            pl.col('small_sell_value').rolling_mean(period).alias("small_sell_value_mean"),
            pl.col('big_sell_value').rolling_mean(period).alias("big_sell_value_mean"),
            (pl.col('small_buy_value') - pl.col('small_sell_value')).alias("level_0_diff_mean").rolling_mean(period),
            (pl.col('big_buy_value') - pl.col('big_sell_value')).alias("level_1_diff_mean").rolling_mean(period),
            pl.col('small_buy_value').rolling_std(period).alias("small_buy_value_std"),
            pl.col('big_buy_value').rolling_std(period).alias("big_buy_value_std"),
            pl.col('small_sell_value').rolling_std(period).alias("small_sell_value_std"),
            pl.col('big_sell_value').rolling_std(period).alias("big_sell_value_std"),
            (pl.col('small_buy_value') - pl.col('small_sell_value')).alias("level_0_diff_std").rolling_std(period),
            (pl.col('big_buy_value') - pl.col('big_sell_value')).alias("level_1_diff_std").rolling_std(period),

            pl.col('small_buy_value').rolling_skew(period).alias("small_buy_value_skew"),
            pl.col('big_buy_value').rolling_skew(period).alias("big_buy_value_skew"),
            pl.col('small_sell_value').rolling_skew(period).alias("small_sell_value_skew"),
            pl.col('big_sell_value').rolling_skew(period).alias("big_sell_value_skew"),
            (pl.col('small_buy_value') - pl.col('small_sell_value')).alias("level_0_diff_skew").rolling_skew(period),
            (pl.col('big_buy_value') - pl.col('big_sell_value')).alias("level_1_diff_skew").rolling_skew(period),
        ])
        small_buy_value_kurt = RollStats(period)
        big_buy_value_kurt = RollStats(period)
        small_sell_value_kurt = RollStats(period)
        big_sell_value_kurt = RollStats(period)
        level_0_diff_kurt = RollStats(period)
        level_1_diff_kurt = RollStats(period)

        datetime = [] 
        small_buy_value_kurt_list = [] 
        big_buy_value_kurt_list = [] 
        small_sell_value_kurt_list = [] 
        big_sell_value_kurt_list = [] 
        level_0_diff_kurt_list = [] 
        level_1_diff_kurt_list = [] 
    
        for j in range(df.shape[0]):
            small_buy_value_kurt.update_raw(df[j,'small_buy_value'])
            big_buy_value_kurt.update_raw(df[j,'big_buy_value'])
            small_sell_value_kurt.update_raw(df[j,'small_sell_value'])
            big_sell_value_kurt.update_raw(df[j,'big_sell_value'])
            level_0_diff_kurt.update_raw(df[j,'small_buy_value'] -df[j,'small_sell_value'])
            level_1_diff_kurt.update_raw(df[j,'big_buy_value']-df[j,'big_sell_value'])
            datetime.append(df[j,"datetime"])
            if not small_buy_value_kurt.initialized():
                small_buy_value_kurt_list.append(np.nan)
                big_buy_value_kurt_list.append(np.nan)
                small_sell_value_kurt_list.append(np.nan)
                big_sell_value_kurt_list.append(np.nan)
                level_0_diff_kurt_list.append(np.nan)
                level_1_diff_kurt_list.append(np.nan)
            else:
                small_buy_value_kurt_list.append(small_buy_value_kurt.kurt())
                big_buy_value_kurt_list.append(big_buy_value_kurt.kurt())
                small_sell_value_kurt_list.append(small_sell_value_kurt.kurt())
                big_sell_value_kurt_list.append(big_sell_value_kurt.kurt())
                level_0_diff_kurt_list.append(level_0_diff_kurt.kurt())
                level_1_diff_kurt_list.append(level_1_diff_kurt.kurt())
        factor2 = pl.DataFrame({
                "small_buy_value_kurt":small_buy_value_kurt_list,
                "big_buy_value_kurt":big_buy_value_kurt_list,
                "small_sell_value_kurt":small_sell_value_kurt_list,
                "big_sell_value_kurt":big_sell_value_kurt_list,
                "level_0_diff_kurt":level_0_diff_kurt_list,
                "level_1_diff_kurt":level_1_diff_kurt_list,
        })
        factor = pl.concat([factor,factor2],how='horizontal')
        corr = corrections(
                    factor,
                    label,
                    corr_type ="pearson",
                )
        #return corr
        print(f"the period {period} factor corrections are {corr}")

from nautilus_trader.indicators.zigzag import Zigzag
import pandas as pd 
def test_zigzag(df):
    label = df.select([pl.col('datetime'),pl.col("label")])
    for change_percent in np.arange(0.005,0.05,0.005):
        zigzag =Zigzag(change_percent,False)
        datetime = [] 
        zigzag_value_list = [] 
        zigzag_strength_list = []   
        zizag_anchored_vwap_list = [] 
        zizag_last_anchored_vwap_list = [] 
        zigzag_volume_ratio_list = [] 
        for j in range(df.shape[0]):
            zigzag.update_raw(df[j,"open"],df[j,"high"],df[j,"low"],df[j,"close"],df[j,"volume"],pd.Timestamp(df[j,"datetime"], tz="UTC"))
            datetime.append(df[j,"datetime"])
            if not zigzag.initialized:
                zigzag_value_list.append(np.nan)
                zigzag_strength_list.append(np.nan)
                zizag_anchored_vwap_list.append(np.nan)
                zizag_last_anchored_vwap_list.append(np.nan)
                zigzag_volume_ratio_list.append(np.nan)
            else:
                if zigzag.zigzag_direction == 1:
                    zigzag_value_list.append((zigzag.high_price-df[j,"close"])/zigzag.length)
                    zigzag_strength_list.append(zigzag.zigzag_direction*zigzag.length/zigzag.low_price)
                else:
                    zigzag_value_list.append((-zigzag.low_price+df[j,"close"])/zigzag.length)
                    zigzag_strength_list.append(zigzag.zigzag_direction*zigzag.length/zigzag.high_price)
                zizag_anchored_vwap_list.append(zigzag.anchored_vwap)
                zizag_last_anchored_vwap_list.append(zigzag.last_anchored_vwap)
                zigzag_volume_ratio_list.append(zigzag.last_sum_volume)

        factor = pd.DataFrame({
            "datetime":datetime,
            "zigzag" : zigzag_value_list,
            "zigzag_strength":zigzag_strength_list,
            "zizag_anchored_vwap":zizag_anchored_vwap_list,
            "zizag_last_anchored_vwap":zizag_last_anchored_vwap_list,
            "zigzag_volume_ratio":zigzag_volume_ratio_list,
        })
        label = df.select([pl.col('datetime'),pl.col("label")])
        corr = corrections(
            factor,
            label,
            corr_type ="pearson",
        )
        print(f"the change_percent {change_percent} factor corrections are {corr}")


from nautilus_trader.indicators.eri import ElderRayIndex
def test_eri(df):
    label = df.select([pl.col('datetime'),pl.col("label")])
    for period in range(50,500,50):
        eri =ElderRayIndex(period)
        datetime = [] 
        eri_bull_list = [] 
        eri_bear_list = [] 
    
        for j in range(df.shape[0]):
            eri.update_raw(df[j,"high"],df[j,"low"],df[j,"close"])
            datetime.append(df[j,"datetime"])
            if not eri.initialized:
                eri_bull_list.append(np.nan)
                eri_bear_list.append(np.nan)
            else:
                eri_bull_list.append(eri.bull)
                eri_bear_list.append(eri.bear)
        factor = pd.DataFrame({
            "datetime":datetime,
            "eri_bull" : eri_bull_list,
            "eri_bear" : eri_bear_list,
        })
        label = df.select([pl.col('datetime'),pl.col("label")])
        corr = corrections(
            factor,
            label,
            corr_type ="pearson",
        )
        print(f"the period {period} factor corrections are {corr}")


from nautilus_trader.indicators.bias import Bias
def test_bias(df):
    label = df.select([pl.col('datetime'),pl.col("label")])
    for period in range(50,500,50):
        bias =Bias(period)
        datetime = [] 
        bias_list = [] 
    
        for j in range(df.shape[0]):
            bias.update_raw(df[j,"close"])
            datetime.append(df[j,"datetime"])
            if not bias.initialized:
                bias_list.append(np.nan)
            else:
                bias_list.append(bias.value)
        factor = pd.DataFrame({
            "datetime":datetime,
            "bias" : bias_list,
        })
        label = df.select([pl.col('datetime'),pl.col("label")])
        corr = corrections(
            factor,
            label,
            corr_type ="pearson",
        )
        print(f"the period {period} factor corrections are {corr}")

from nautilus_trader.indictaors.bop import BalanceOfPower 
def test_bias(df):
    label = df.select([pl.col('datetime'),pl.col("label")])
    bias =Bias()
    datetime = [] 
    bias_list = [] 

    for j in range(df.shape[0]):
        bias.update_raw(df[j,"open"],df[j,"high"],df[j,"low"],df[j,"close"])
        datetime.append(df[j,"datetime"])
        if not bias.initialized:
            bias_list.append(np.nan)
        else:
            bias_list.append(bias.value)
    factor = pd.DataFrame({
        "datetime":datetime,
        "bias" : bias_list,
    })
    label = df.select([pl.col('datetime'),pl.col("label")])
    corr = corrections(
        factor,
        label,
        corr_type ="pearson",
    )
    print(f"the factor corrections are {corr}")


from nautilus_trader.indicators.rsi import RelativeStrengthIndex

def test_rsi(df):
    label = df.select([pl.col('datetime'),pl.col("label")])
    for period in range(500,3000,100):
        rsi=RelativeStrengthIndex(period)
        datetime = [] 
        rsi_list = [] 
    
        for j in range(df.shape[0]):
            rsi.update_raw(df[j,"close"])
            datetime.append(df[j,"datetime"])
            if not rsi.initialized:
                rsi_list.append(np.nan)
            else:
                rsi_list.append(rsi.value)
        factor = pd.DataFrame({
            "datetime":datetime,
            "rsi" : rsi_list,
        })
        label = df.select([pl.col('datetime'),pl.col("label")])
        corr = corrections(
            factor,
            label,
            corr_type ="pearson",
        )
        return corr


from nautilus_trader.indicators.vortex import Vortex
def test_vortex(df):
    label = df.select([pl.col('datetime'),pl.col("label")])
    for pvortexod in range(50,500,50):
        vortex =Vortex(pvortexod)
        datetime = [] 
        vortex_vip_list = [] 
        vortex_vim_list = [] 
    
        for j in range(df.shape[0]):
            vortex.update_raw(df[j,"high"],df[j,"low"],df[j,"close"])
            datetime.append(df[j,"datetime"])
            if not vortex.initialized:
                vortex_vip_list.append(np.nan)
                vortex_vim_list.append(np.nan)
            else:
                vortex_vip_list.append(vortex.vip)
                vortex_vim_list.append(vortex.vim)
        factor = pd.DataFrame({
            "datetime":datetime,
            "vortex_vip" : vortex_vip_list,
            "vortex_vim" : vortex_vim_list,
        })
        label = df.select([pl.col('datetime'),pl.col("label")])
        corr = corrections(
            factor,
            label,
            corr_type ="pearson",
        )
        print(f"the pvortexod {pvortexod} factor corrections are {corr}") 

from nautilus_trader.indicators.linear_regression import LinearRegression
def test_cfo(df):
    label = df.select([pl.col('datetime'),pl.col("label")])
    for period in range(50,500,50):
        cfo = LinearRegression(period)
        datetime = [] 
        slope_list = [] 
        intercept_list = [] 
        #degree_list = [] 
        cfo_list = [] 
        R2_list = [] 
        value_list = [] 
    
        for j in range(df.shape[0]):
            cfo.update_raw(df[j,"close"])
            datetime.append(df[j,"datetime"])
            if not cfo.initialized:
                cfo_list.append(np.nan)
                slope_list.append(np.nan)
                intercept_list.append(np.nan)
                R2_list.append(np.nan)
                value_list.append(np.nan)
            else:
                cfo_list.append(cfo.cfo)
                slope_list.append(cfo.slope)
                intercept_list.append(cfo.intercept)
                R2_list.append(cfo.R2)
                value_list.append(cfo.value)
        factor = pd.DataFrame({
            "datetime":datetime,
            "cfo" : cfo_list,
            "slope":slope_list,
            "intercept":intercept_list,
            "R2":R2_list,
            "value":value_list,
        })
        label = df.select([pl.col('datetime'),pl.col("label")])
        corr = corrections(
            factor,
            label,
            corr_type ="pearson",
        )
        print(f"the period {period} factor corrections are {corr}")
