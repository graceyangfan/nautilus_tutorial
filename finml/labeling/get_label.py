
import polars as pl 
import numpy as np 


def calc_change_since_pivot(current,last_pivot):
    if(last_pivot == 0): last_pivot = 1 ** (-100) # avoid division by 0
    perc_change_since_pivot = (current - last_pivot) / abs(last_pivot)
    return perc_change_since_pivot

def get_zigzag(idx, row, taip=None):
    return {
        "datetime": row[0],
        "value": row[1],
        "type": taip,
        "idx":idx,
    }

def create_label(
    df,
    threshold = 0.01,
    stop_loss = None,
    cut_label = True,
):
    zigzags = []
    for idx,item in enumerate(df.select(["datetime","close"]).iterrows()):
        is_starting = (idx == 0)
        if is_starting:
            zigzags.append(get_zigzag(idx,item))
            continue  

        is_first_line = (len(zigzags) == 1) 
        if is_first_line:
            perc_change_since_pivot = calc_change_since_pivot(item[-1],zigzags[-1]["value"])
            if abs(perc_change_since_pivot) >= threshold:
                if perc_change_since_pivot > 0:
                    zigzags.append(get_zigzag(idx, item,"Peak"))
                    zigzags[0]["type"] = "Through"
                else:
                    zigzags.append(get_zigzag(idx, item, "Trough"))
                    zigzags[0]["type"] = "Peak" 
            continue 
        is_through = zigzags[-2]["value"] > zigzags[-1]["value"]
        is_ending = (idx == df.shape[0] - 1)
        last_pivot = float(zigzags[-1]["value"])
        # based on last pivot type, look for reversal or continuation
        if(is_through):
            perc_change_since_pivot = calc_change_since_pivot(item[-1],zigzags[-1]["value"])
            is_reversing = (perc_change_since_pivot >= threshold) or is_ending
            is_continuing = item[-1] <= last_pivot
            if (is_continuing): 
                zigzags[-1] = get_zigzag(idx,item, "Trough")
            elif (is_reversing): 
                zigzags.append(get_zigzag(idx,item, "Peak"))
        else:
            perc_change_since_pivot = calc_change_since_pivot(item[-1],zigzags[-1]["value"])
            is_reversing = (perc_change_since_pivot <= -threshold) or is_ending
            is_continuing = item[-1] >= last_pivot
            if(is_continuing): 
                zigzags[-1] = get_zigzag(idx,item, "Peak")
            elif (is_reversing): 
                zigzags.append(get_zigzag(idx,item, "Trough"))

    zigzags = pl.DataFrame(zigzags)
    zigzags = zigzags.select([
        pl.all(),
        pl.col("datetime").shift(-1).alias("event_ends"),
        pl.col("value").shift(-1).alias("prevext")
    ])
    assert zigzags.shape[0] >=2 
    df = df.join(zigzags, on = "datetime", how = "left")
    df = df.select(
        [pl.col(item).fill_null(strategy = "forward") if item in ["prevext","event_ends"] else pl.col(item) for item in df.columns]
    )
    df = df.select(
        [pl.all(), (pl.col("prevext")/pl.col("close") - 1.0).alias("label")]
    )
    correct_label = [] 
    if stop_loss:
        total_returns = df.select("label").to_numpy().flatten() 
        return_list = df.select(
            ((pl.col("close").shift(-1) - pl.col("close"))/pl.col("close")).alias("return")
        )
        return_list = return_list.filter(pl.col("return").is_not_null()).to_numpy().flatten() 

        close_array = df.select("close").to_numpy().flatten() 

        for i in range(zigzags.shape[0]-1):
            start_idx = zigzags[i,"idx"]
            end_idx = zigzags[i+1,"idx"]
            for j in range(start_idx,end_idx):
                local_returns = return_list[j:min(end_idx + 1,len(return_list))]
                min_acc = 0 
                if total_returns[j] > 0:
                    min_acc = min((min(close_array[j+1:end_idx+1])-close_array[j])/close_array[j],0)
                else:
                    min_acc = max((max(close_array[j+1:end_idx+1])-close_array[j])/close_array[j],0)
                if total_returns[j] > 0:
                    if min_acc > -stop_loss:
                        correct_label.append(total_returns[j])
                    else:
                        correct_label.append(min_acc)
                else:
                    if min_acc < stop_loss:
                        correct_label.append(total_returns[j])
                    else:
                        correct_label.append(min_acc)
        #replace label of df 
        df = df[:len(correct_label),:]
        df.replace("label",pl.Series(correct_label))

    ## drop the front and the last trend data because zigzag is meanless on these data 
    df = df.filter((pl.col("datetime")>=zigzags[1,"datetime"])&(pl.col("datetime")<=(zigzags[-2,"datetime"])))

    df = df.select(
        [
            pl.col("datetime"),
            pl.col("close"),
            pl.col("datetime").alias("event_starts"),
            pl.col("event_ends"),
            pl.col("label")
        ]
    )
    if cut_label:
        label_array = df[:,"label"].to_numpy()
        df = df.select([
            pl.all().exclude("label"),
            (pl.when(pl.col("label")>label_array.mean() +5.0*label_array.std())
             .then(label_array.mean() +5.0*label_array.std())
             .otherwise(pl.col("label"))).alias("label"),
        ])
        df = df.select([
            pl.all().exclude("label"),
            (pl.when(pl.col("label")< label_array.mean() - 5.0*label_array.std())
             .then(label_array.mean() - 5.0*label_array.std())
             .otherwise(pl.col("label"))).alias("label"),
        ])

    return df 

