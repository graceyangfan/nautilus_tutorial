import glob 
import polars as pl 
import pandas as pd 

def create_label(df,threshold,stop_loss):
    zigzags = []
    def calc_change_since_pivot(row, key):
        current = row[key]
        last_pivot = zigzags[-1]["Value"]
        if(last_pivot == 0): last_pivot = 1 ** (-100) # avoid division by 0
        perc_change_since_pivot = (current - last_pivot) / abs(last_pivot)
        return perc_change_since_pivot

    def get_zigzag(idx,row, taip=None):
        if(taip == "Peak"): key = "high"
        elif(taip == "Trough"): key = "low"
        else: key = "close"

        return {
            "Time": row["datetime"],
            "Value": row[key],
            "Type": taip,
            "idx":idx,
        }
    
    for ix, row in df.iterrows():
         # handle first point
        is_starting = ix == 0
        if(is_starting):
            zigzags.append(get_zigzag(ix,row))
            continue

        # handle first line
        is_first_line = len(zigzags) == 1
        if(is_first_line):
            perc_change_since_pivot = calc_change_since_pivot(row, "close")

            if(abs(perc_change_since_pivot) >= threshold):
                if(perc_change_since_pivot > 0):
                    zigzags.append(get_zigzag(ix,row, "Peak"))
                    zigzags[0]["Type"] = "Trough"
                else: 
                    zigzags.append(get_zigzag(ix,row, "Trough"))
                    zigzags[0]["Type"] = "Peak"
            continue
        # handle other lines
        is_trough = zigzags[-2]["Value"] > zigzags[-1]["Value"]
        is_ending = ix == len(df.index) - 1
        last_pivot = float(zigzags[-1]["Value"])
        # based on last pivot type, look for reversal or continuation
        if(is_trough):
            perc_change_since_pivot = calc_change_since_pivot(row, "close")
            is_reversing = (perc_change_since_pivot >= threshold) or is_ending
            is_continuing = row["close"] <= last_pivot
            if (is_continuing): 
                zigzags[-1] = get_zigzag(ix,row, "Trough")
            elif (is_reversing): 
                zigzags.append(get_zigzag(ix,row, "Peak"))
        else:
            perc_change_since_pivot = calc_change_since_pivot(row, "close")
            is_reversing = (perc_change_since_pivot <= -threshold) or is_ending
            is_continuing = row["close"] >= last_pivot
            if(is_continuing): 
                zigzags[-1] = get_zigzag(ix,row, "Peak")
            elif (is_reversing): 
                zigzags.append(get_zigzag(ix,row, "Trough"))
    zigzags = pd.DataFrame(zigzags)
    zigzags["PrevExt"] = zigzags.Value.shift(-1)
    df=zigzags.merge(df,left_on="Time",right_on="datetime",how="right")
    df["Type"]= df["Type"].map({"Trough":1,"Peak":2})
    df["Type"]=df["Type"].replace(np.nan,0)
    df["PrevExt"] = df["PrevExt"].fillna(method='ffill')
    df["target"] = df["PrevExt"]/df["close"]

    total_returns = df["target"].values - 1 
    returns = (df.close.shift(-1)-df.close)/df.close 
    returns_list = returns.values
    close_array = df.close.values
    high_array = df.high.values
    low_array = df.low.values
    df_label = []
    for i in range(len(zigzags)-1):
        st_idx = zigzags.loc[i,"idx"]
        ed_idx = zigzags.loc[i+1,"idx"]
        for j in range(st_idx,ed_idx):
            local_returns = returns_list[j:ed_idx+1]
            min_acc = 0 
            if total_returns[j] > 0:
                min_acc = min((min(low_array[j+1:ed_idx+1])-close_array[j])/close_array[j],0)
            else:
                min_acc = max((max(high_array[j+1:ed_idx+1])-close_array[j])/close_array[j],0)
            if total_returns[j] > 0:
                if min_acc > -stop_loss:
                    df_label.append( total_returns[j])
                else:
                    df_label.append( min_acc)
            else:
                if min_acc < stop_loss:
                    df_label.append( total_returns[j])
                else:
                    df_label.append( min_acc)

    df = df.iloc[:len(df_label)]
    df["label"] = df_label

    return df 


def read_data(
    minute,
    threshold =0.04,
    stop_loss =0.02,
):
    df =pd.DataFrame([])
    files = glob.glob(f"../train/*-PERP.BINANCE-{minute}-MINUTE-LAST-EXTERNAL_base.parquet")
    for filename in files:
        df_in = pd.read_parquet(filename)
        df_in = df_in.reset_index()
        df_in = create_label(df_in,threshold,stop_loss)
        df_in = df_in[(df_in["long_condition"]>0.0)  | \
                      (df_in["short_condition"]>0.0)]
        df = pd.concat([df,df_in])
    df.to_parquet("example.parquet")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbol', default='BTCUSDT')
    parser.add_argument("--venue",default='BINANCE') 
    parser.add_argument("--minute",default=5,type=int) 
    parser.add_argument("--direction",default=-1,type=int) 
    args = parser.parse_args()
    instrument_id = f"{args.symbol}-PERP.{args.venue}"
    bar_type = instrument_id + '-'+str(args.minute) +"-MINUTE-LAST-EXTERNAL"
    read_data(args.minute,0.04,0.02)

    df =pd.read_parquet("example.parquet")
    df["o_label"] = df["label"]
    if args.direction  == 1:
        up_df = df[(df["long_condition"]>0)]
        up_df = up_df.sample(frac=1)
        up_df["label"] = up_df["o_label"]>= 0.02
    elif args.direction  == -1:
        up_df = df[df["short_condition"]>0]
        up_df = up_df.sample(frac=1)
        up_df["label"] = up_df["o_label"]<= -0.02
    up_df["label"] = up_df["label"].astype(int)

    