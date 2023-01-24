
from features.features import FracDiff,Entropy,MicroStructucture
import numpy as np 
import polars as pl
import pandas as pd 
from finml.utils.stats import corrections 

def ms_feature_evaluate(df, labeled_df):
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
            "vpin" : vpin
        })
        label = labeled_df.select([pl.col('datetime'),pl.col("label")])
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


def test_diff(df, labeled_df):
    label = labeled_df.select([pl.col('datetime'),pl.col("label")])
    for order in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
        for period in range(50,500,50):
            frac_dif = FracDiff(order,period)
            value = [] 
            datetime = [] 
            for j in range(df.shape[0]):
                    frac_dif.update_raw(df[j,"close"])
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


def test_value_diff(df, labeled_df):
    label = labeled_df.select([pl.col('datetime'),pl.col("label")])
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