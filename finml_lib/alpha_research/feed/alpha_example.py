import polars as pl
from expr_engine import ExprEngine

def parse_config_to_fields():
    """create factors from config"""
    f_return = "($close/Ref($close, 1)-1)"
    f_adv5 = "Mean($money, 5)"
    f_adv10 = "Mean($money, 10)"
    f_adv15 = "Mean($money, 15)"
    f_adv20 = "Mean($money, 20)"
    f_adv30 = "Mean($money, 30)"
    f_adv40 = "Mean($money, 40)"
    f_adv50 = "Mean($money, 50)"
    f_adv60 = "Mean($money, 60)"
    f_adv120 = "Mean($money, 120)"
    f_adv180 = "Mean($money, 180)"

    alpha_components = {
        "alpha001": f"CSRank(IdxMax(Power(If({f_return}<0, Std({f_return}, 20), $close), 2), 5))-0.5",
        "alpha002": "-1*Corr(CSRank(Delta(Log($volume), 2)), CSRank(($close-$open)/$open), 6)",
        "alpha003": "-1*Corr(CSRank($open), CSRank($volume), 10)",
        "alpha004": "-1*Rank(CSRank($low), 9)",
        "alpha005": f"CSRank($open-Sum($vwap, 10)/10)*(-1*CSRank($close-$vwap))",
        "alpha006": "-1*Corr($open, $volume, 10)",
        "alpha007": f"If({f_adv20}<$volume, 0-Rank(Abs(Delta($close, 7)), 60) * Sign(Delta($close, 7)), -1)",
        "alpha008": f"-1*CSRank(Sum($open, 5) * Sum({f_return}, 5) - Ref(Sum($open, 5) * Sum({f_return}, 5), 10))",
        "alpha009": f"If(0 < Min(Delta($close, 1), 5), Delta($close, 1), If(Max(Delta($close, 1), 5) < 0, Delta($close, 1), -1*Delta($close, 1)))",
        "alpha010": f"CSRank(If(0 < Min(Delta($close, 1), 4), Delta($close, 1), If(Max(Delta($close, 1), 4) < 0, Delta($close, 1), -1*Delta($close, 1))))",
        "alpha011": f"(CSRank(Max($vwap-$close, 3)) + CSRank(Min($vwap-$close, 3))) * CSRank(Delta($volume, 3))",
        "alpha012": f"Sign(Delta($volume, 1)) * (-1 * Delta($close, 1))",
        "alpha013": "-1*CSRank(Cov(CSRank($close), CSRank($volume), 5))",
        "alpha014": f"-1*CSRank(Delta({f_return}, 3))*Corr($open, $volume, 10)",
        "alpha015": "-1*Sum(CSRank(Corr(CSRank($high), CSRank($volume), 3)), 3)",
        "alpha016": "-1*CSRank(Cov(CSRank($high), CSRank($volume), 5))",
        "alpha017": f"-1*CSRank(Rank($close, 10))*CSRank(Delta(Delta($close, 1), 1))*CSRank(Rank($volume/{f_adv20}, 5))",
        "alpha018": f"-1*CSRank(Std(Abs($close-$open), 5) + ($close-$open) + Corr($close, $open, 10))",
        "alpha019": f"-1*Sign((($close - Ref($close, 7)) + Delta($close, 7)))*(1+CSRank(1+Sum({f_return}, 250)))",
        "alpha020": "-1*CSRank($open-Ref($high, 1))*CSRank($open-Ref($close, 1))*CSRank($open-Ref($low, 1))",
        "alpha021": f"If(Mean($close, 8) + Std($close, 8) < Mean($close, 2), -1, If(Mean($close, 2) < Mean($close, 8) - Std($close, 8), 1, If($volume <= {f_adv20}, 1, -1)))",
        "alpha022": "-1*Delta(Corr($high, $volume, 5), 5)*CSRank(Std($close, 20))",
        "alpha023": "If(Mean($high, 20)<$high, -1*Delta($high, 2), 0)",
        "alpha024": "If((Delta(Mean($close, 100) , 100) / Ref($close, 100)) <= 0.05, (Min($close,100)-$close),  -1*Delta($close, 3))",
        "alpha025": f"CSRank(-1*{f_return}*{f_adv20}*$vwap*($high-$close))",
        "alpha026": "0-Max(Corr(Rank($volume, 5), Rank($high, 5), 5), 3)",
        "alpha027": "If(0.5<CSRank(Mean(Corr(CSRank($volume), CSRank($vwap), 6), 2)), -1, 1)",
        "alpha028": f"CSScale((Corr({f_adv20}, $low, 5) + (($high + $low) / 2)) - $close)",
        "alpha029": f"Min(Prod(CSRank(Sum(Min(CSRank(-Delta($close-1,5)), 2), 1)),1), 5) + Rank(Ref(-1*{f_return}, 6), 5)",
        "alpha030": "(1.0-CSRank(Sign($close - Ref($close, 1))+Sign(Ref($close, 1) - Ref($close, 2))+Sign(Ref($close, 2) - Ref($close, 3)))) * Sum($volume, 5) / Sum($volume, 20)",
        "alpha031": f"CSRank(WMA(-CSRank(Delta($close, 10)), 10)) + CSRank(-Delta($close, 3)) + Sign(Corr({f_adv20}, $low, 12))",
        "alpha032": "CSScale(Mean($close, 7)-$close) + 20*CSScale(Corr($vwap, Ref($close, 5), 230))",
        "alpha033": "CSRank($close / ($close - $open))",
        "alpha034": f"CSRank(2 - CSRank(Std({f_return}, 2) / Std({f_return}, 5)) - CSRank(Delta($close, 1)))",
        "alpha035": f"Rank($volume, 32) * (1 - Rank(($close + $high) - $low, 16)) * (1 - Rank({f_return}, 32))",
        "alpha036": f"2.21*CSRank(Corr($close - $open, Ref($volume, 1), 15))+0.7*CSRank($open-$close)+0.73*CSRank(Rank(Ref(-1*{f_return}, 6), 5))+"
        f"CSRank(Abs(Corr($vwap, {f_adv20}, 6)))+0.6*CSRank((Mean($close, 200)-$open)*($close-$open))",
        "alpha037": "CSRank(Corr(Ref($open-$close, 1), $close, 200))+CSRank($open-$close)",
        "alpha038": "-CSRank(Rank($close, 10))*CSRank($close/$open)",
        "alpha039": f"-CSRank(Delta($close, 7) * (1-CSRank(WMA($volume / {f_adv20}, 9)))) * (1+CSRank(Sum({f_return},250)))",
        "alpha040": "-CSRank(Std($high, 10))*Corr($high, $volume, 10)",
        "alpha041": "Power($high*$low, 0.5)-$vwap",
        "alpha042": "CSRank($vwap-$close)/CSRank($vwap+$close)",
        "alpha043": f"Rank($volume / {f_adv20}, 20) * Rank(-1*Delta($close, 7), 8)",
        "alpha044": "-1*Corr($high, CSRank($volume), 5)",
        "alpha045": "-CSRank(Mean(Ref($close, 5), 20))*Corr($close, $volume, 2)*CSRank(Corr(Sum($close, 5), Sum($close, 20), 2))",
        "alpha046": "If(0.25 < (Ref($close, 20) - Ref($close, 10)) / 10 - (Ref($close, 10)-$close) / 10, -1, If((Ref($close, 20) "
        "- Ref($close, 10)) / 10 - (Ref($close, 10) - $close) / 10 < 0, 1, -1 * ($close - Ref($close, 1))))",
        "alpha047": f"CSRank(1/$close)*CSRank($high-$close)*$volume/{f_adv20}*$high/Mean($high, 5)-CSRank($vwap - Ref($vwap, 5))",
        # # 'alpha048': use  indneutralize
        "alpha049": "If((Ref($close, 20) - Ref($close, 10)) / 10 - (Ref($close, 10) - $close) / 10 < -0.1, 1, -1*($close - Ref($close, 1)))",
        "alpha050": f"-1*Max(CSRank(Corr(CSRank($volume), CSRank($vwap), 5)), 5)",
        "alpha051": "If((Ref($close, 20) - Ref($close, 10)) / 10 - (Ref($close, 10) - $close) / 10 < -0.05, 1, -1*($close - Ref($close, 1)))",
        "alpha052": f"(Ref(Min($low, 5), 5)-Min($low, 5))*CSRank((Sum({f_return}, 240) - Sum({f_return}, 20)) / 220)*Rank($volume, 5)",
        "alpha053": "-1*Delta((($close - $low) - ($high - $close)) / ($close*1.000001 - $low), 9)",
        "alpha054": "-1*($low - $close) * Power($open, 5) / (($low*1.000001 - $high) * Power($close, 5))",
        "alpha055": "-1*Corr(CSRank(($close - Min($low, 12)) / (Max($high, 12) - Min($low, 12))), CSRank($volume), 6)",
        "alpha056": f"-1*CSRank(Sum({f_return}, 10)/Sum(Sum({f_return}, 2), 3))*CSRank({f_return}*$open_interest)",
        "alpha057": f"-1*($close-$vwap)/WMA(CSRank(IdxMax($close, 30)), 2)",
        # # 'alpha058': use  indneutralize
        # # 'alpha059': use  indneutralize
        "alpha060": "CSScale(CSRank(IdxMax($close, 10))) - 2*CSScale(CSRank(((($close - $low) - ($high - $close)) / ($high - $low)) * $volume))",
        "alpha061": f"If(CSRank($vwap - Min($vwap, 16)) < CSRank(Corr($vwap, {f_adv180}, 17)), 1, 0)",
        "alpha062": f"If(CSRank(Corr($vwap, Sum({f_adv20}, 22), 9)) < CSRank(If(2*CSRank($open) < CSRank(($high+$low)/2) + CSRank($high), 1, 0)), -1, 0)",
        # # 'alpha063': use  indneutralize
        "alpha064": f"If(CSRank(Corr(Sum((($open * 0.178404) + ($low * (1 - 0.178404))), 12), Sum({f_adv120}, 12), 16))"
        f"<CSRank(Delta((((($high + $low) / 2) * 0.178404) + ($vwap * (1 - 0.178404))), 3)), -1, 0)",
        "alpha065": f"If(CSRank(Corr(($open * 0.00817205) + ($vwap * (1 - 0.00817205)), Sum({f_adv60}, 8), 6))<CSRank($open - Min($open, 13)), -1, 0)",
        "alpha066": f"-(CSRank(WMA(Delta($vwap, 3), 7))+Rank(WMA(($low * 0.96633 + $low * (1 - 0.96633) - $vwap) / ($open - ($high + $low) / 2), 11), 6))",
        # # 'alpha067': use  indneutralize
        "alpha068": f"If(Rank(Corr(CSRank($high), CSRank({f_adv15}), 9), 14)<CSRank(Delta($close * 0.518371 + $low * (1 - 0.518371), 1)), -1, 0)",
        # # 'alpha069': use  indneutralize
        # # 'alpha070': use  indneutralize
        "alpha071": f"Greater(Rank(WMA(Corr(Rank($close, 3), Rank({f_adv180},12), 18), 4), 15), Rank(WMA(Power(CSRank($low + $open - 2*$vwap), 2), 16), 4))",
        "alpha072": f"CSRank(WMA(Corr(($high + $low) / 2, {f_adv40}, 8), 10))/CSRank(WMA(Corr(Rank($vwap, 3), Rank($volume, 18), 6), 2))",
        "alpha073": f"-Greater(CSRank(WMA(Delta($vwap, 4), 3)), Rank(WMA(-1*(Delta($open * 0.147155 + $low * (1 - 0.147155), 2) / ($open *0.147155 + $low * (1 - 0.147155))), 3), 16))",
        "alpha074": f"If(CSRank(Corr($close, Sum({f_adv30}, 37), 15))<CSRank(Corr(CSRank($high * 0.0261661 + $vwap * (1 - 0.0261661)), CSRank($volume), 11)), -1, 0)",
        "alpha075": f"If(CSRank(Corr($vwap, $volume, 4))<CSRank(Corr(CSRank($low), CSRank({f_adv50}), 12)), 1, 0)",
        # # 'alpha076': use  indneutralize
        "alpha077": f"Less(CSRank(WMA(($high + $low) / 2 - $vwap, 20)), CSRank(WMA(Corr(($high + $low) / 2, {f_adv40}, 3), 5)))",
        "alpha078": f"Power(CSRank(Corr(Sum($low * 0.352233 + $vwap * (1 - 0.352233), 19), Sum({f_adv40}, 19), 6)), CSRank(Corr(CSRank($vwap), CSRank($volume), 6)))",
        # # 'alpha079': use  indneutralize
        # # 'alpha080': use  indneutralize
        "alpha081": f"If(CSRank(Sum(Log(CSRank(Power(CSRank(Corr($vwap, Sum({f_adv10}, 49), 8)), 4))), 15))<CSRank(Corr(CSRank($vwap), CSRank($volume), 5)), -1, 0)",
        # # 'alpha082': use  indneutralize
        "alpha083": "CSRank(Ref(($high - $low) / Mean($close, 5), 2))*CSRank($volume)*Mean($close, 5) * ($vwap - $close) / ($high*1.0000001-$low)",
        # "alpha084": f"Power(Rank($vwap - Max($vwap, 15), 20), Delta($close, 4))", # meaning less
        "alpha085": f"Power(CSRank(Corr(($high * 0.876703) + ($close * (1 - 0.876703)), {f_adv30}, 9)), CSRank(Corr(Rank(($high + $low) / 2, 3), Rank($volume, 10),7)))",
        "alpha086": f"If(Rank(Corr($close, Sum({f_adv20}, 14), 6), 20)<CSRank(($open+ $close) - ($vwap + $open)), -1, 0)",
        # 'alpha087': use  indneutralize
        "alpha088": f"Less(CSRank(WMA((CSRank($open)+CSRank($low))-(CSRank($high)+CSRank($close)), 8)), Rank(WMA(Corr(Rank($close,8), Rank({f_adv60},20), 8), 6), 2))",
        # # 'alpha089': use  indneutralize
        # # 'alpha090': use  indneutralize
        # # 'alpha091': use  indneutralize
        "alpha092": f"Less(Rank(WMA(If((((($high + $low) / 2) + $close) < ($low + $open)), 1, 0), 14), 18), Rank(WMA(Corr(CSRank($low), CSRank({f_adv30}), 8), 7), 7))",
        # # 'alpha093': use  indneutralize
        "alpha094": f"-Power(CSRank($vwap - Min($vwap, 11)), Rank(Corr(Rank($vwap,19), Rank({f_adv60}, 4), 18), 2))",
        "alpha095": f"If(CSRank($open - Min($open, 12))<Rank(CSRank(Corr(Sum(($high + $low)/ 2, 19), Sum({f_adv40}, 19), 12)), 12), 1, 0)",
        "alpha096": f"-Greater(Rank(WMA(Corr(CSRank($vwap), CSRank($volume), 4), 4), 8), Rank(WMA(IdxMax(Corr(Rank($close, 7), Rank({f_adv60}, 4), 3), 12), 14), 13))",
        # # 'alpha097': use  indneutralize
        "alpha098": f"CSRank(WMA(Corr($vwap, Sum({f_adv5}, 26), 4), 7))-CSRank(WMA(Rank(IdxMin(Corr(CSRank($open), CSRank({f_adv15}), 21), 9), 7), 8))",
        "alpha099": f"If(CSRank(Corr(Sum(($high + $low) / 2, 19), Sum({f_adv60}, 19), 8))<CSRank(Corr($low, $volume, 6)), -1, 0)",
        # # 'alpha100': use  indneutralize
        "alpha101": "(($close - $open) / (($high - $low) + 0.001))",
    }

    return list(alpha_components.values()), list(alpha_components.keys())


def alphaeval(df):
    expr = ExprEngine()
    expr.init()

    fields,names = parse_config_to_fields()

    for name, field in zip(names,fields):
        exp = expr.get_expression(filed)
        se = exp.batch_update(df)
        df= df.with_columns(name = se)

    return df 


if __name__ == "__main__":
    df = pl.read_parquet("crypto_example.parquet")
    
    