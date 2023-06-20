import polars as pl
from  graph import ScatterGraph, SubplotsGraph, BarGraph, HeatmapGraph
from  utils import guess_plotly_rangebreaks

def _group_return(
    pred_label: pl.DataFrame = None, 
    analysis_column = "score",
    reverse: bool = False, 
    N: int = 5,
    **kwargs
) -> tuple:
    if reverse:
        pred_label = pred_label.with_columns([
            (-pl.col(analysis_column)).alias(analysis_column),
        ])

    pred_label = pred_label.sort(by=analysis_column, descending = True)

    # Group1 ~ Group5 only consider the dropna values
    pred_label_drop = pred_label.drop_nulls(subset=[analysis_column])

    # Group
    t_df = pred_label_drop.groupby("datetime").agg([pl.col("label").mean().alias("average")])
    for i in range(N):
        t_df = t_df.with_columns([
            pl.Series(name=f"Group{i+1}", values = pred_label_drop.groupby("datetime").agg(
                [pl.col("label").slice(pl.count()//N * i, pl.count()//N * (i+1)).mean()]
            ).to_numpy().flatten())
        ])
    
    #Long-Short,Long-Average
    t_df = t_df.with_columns([
        (pl.col("Group1") - pl.col(f"Group{N}")).alias("long-short"),
        (pl.col("Group1") - pl.col("average")).alias("long-average"),
    ])
    
    t_df = t_df.drop_nulls()  # for days which does not contain label
    # Cumulative Return By Group
    group_scatter_figure = ScatterGraph(
        t_df.cumsum(),
        layout=dict(
            title="Cumulative Return",
            xaxis=dict(tickangle=45, rangebreaks=kwargs.get("rangebreaks", guess_plotly_rangebreaks(t_df["datetime"]))),
        ),
    ).figure

    t_df = t_df.select(["long-short", "long-average"])
    _bin_size = float(((t_df.max() - t_df.min()) / 20).min())
    group_hist_figure = SubplotsGraph(
        t_df,
        kind_map=dict(kind="DistplotGraph", kwargs=dict(bin_size=_bin_size)),
        subplots_kwargs=dict(
            rows=1,
            cols=2,
            print_grid=False,
            subplot_titles=["long-short", "long-average"],
        ),
    ).figure

    return group_scatter_figure, group_hist_figure