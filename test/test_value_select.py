def select_params(df,njobs=1):
    def compute_value(params):
        newbars = zscore_filter(params["df"],params["period"].item(),params["threshold"],params["value"])
        if len(newbars) < 1000:
            return (1,-1)
        labeled_df =create_label(
            newbars.select([
            pl.col("ts_event").alias("datetime"),
            pl.col("close"),
            pl.col("buyer_maker_imbalance")]),
            0.01,
            0.005
        )

        labeled_df = labeled_df.with_columns([
            (pl.when(pl.col("buyer_maker_imbalance")>pl.col("buyer_maker_imbalance").mean() +5.0*pl.col("buyer_maker_imbalance").std())
             .then(pl.col("buyer_maker_imbalance").mean() +5.0*pl.col("buyer_maker_imbalance").std())
             .otherwise(pl.col("buyer_maker_imbalance"))).alias("buyer_maker_imbalance"),
        ])
        labeled_df = labeled_df.with_columns([
            (pl.when(pl.col("buyer_maker_imbalance")< pl.col("buyer_maker_imbalance").mean() - 5.0*pl.col("buyer_maker_imbalance").std())
             .then(pl.col("buyer_maker_imbalance").mean() - 5.0*pl.col("buyer_maker_imbalance").std())
             .otherwise(pl.col("buyer_maker_imbalance"))).alias("buyer_maker_imbalance"),
        ])
        
        #normal 
        labeled_df = labeled_df.with_columns([
            ((pl.col("buyer_maker_imbalance")-pl.col("buyer_maker_imbalance").mean())/pl.col("buyer_maker_imbalance").std()).alias("buyer_maker_imbalance"),
            ((pl.col("label")-pl.col("label").mean())/pl.col("label").std()).alias("label")
        ])
    
        
        in_sample_corr,out_sample_corr = test_imbalance_corr(labeled_df)  
        if np.isnan(in_sample_corr) or np.isnan(out_sample_corr):
            return (1,-1)
        if np.sign(in_sample_corr)!=np.sign(out_sample_corr):
            return (1,-1)
        timeduring = newbars.select(pl.col("ts_event")-pl.col("ts_init")).median().row(0)[0] 
        return in_sample_corr,out_sample_corr,timeduring,{"period":period,"threshold":threshold,"value":value}
    
    values = (df["price"]*df["quantity"])
    params=[]
    for period in np.arange(50,100,50):
        for threshold in np.arange(3,15,3):
            for value in np.linspace(0,int(values.quantile(0.9)),10)[1:]:
                params.append({
                    "df":df,
                    "period":period,
                    "threshold":threshold,
                    "value":value
                })
    results = Parallel(n_jobs=njobs)(delayed(compute_value)(param) for param in tqdm(params))
    params = [] 
    in_sample_corr_list = [] 
    out_sample_corr_list = [] 
    timedurings = [] 
    for idx,item in enumerate(results):
        if np.sign(item[0]) == np.sign(item[1]):
            params.append(item[-1])
            in_sample_corr_list.append(item[0])
            out_sample_corr_list.append(item[1])
            timedurings.append(item[2])
    results = pd.DataFrame({
        "params":params,
        "in_sample_corr":in_sample_corr_list,
        "out_sample_corr":out_sample_corr_list,
        "timedurings":timedurings,
    })
    results = results.sort_values("out_sample_corr",ascending=False)
    return results
