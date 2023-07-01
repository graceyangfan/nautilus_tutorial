import polars as pl
import statsmodels.api as sm
import plotly.graph_objs as go
import matplotlib.pyplot as plt
from scipy import stats
from typing import Sequence, Literal
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
    expressions = [pl.col("label").mean().alias("average")]
    for i in range(N):
        expressions.append(
            pl.col("label").slice(pl.count()//N * i, pl.count()//N * (i+1)).mean().alias(f"Group{i+1}")
        )
    t_df = pred_label_drop.groupby("datetime").agg(expressions).sort("datetime")

    #Long-Short,Long-Average
    t_df = t_df.with_columns([
        (pl.col("Group1") - pl.col(f"Group{N}")).alias("long-short"),
        (pl.col("Group1") - pl.col("average")).alias("long-average"),
    ])
    
    t_df = t_df.drop_nulls()  # for days which does not contain label
    # Cumulative Return By Group
    group_scatter_figure = ScatterGraph(
        t_df.select([pl.col("datetime"),pl.all().exclude("datetime").cumsum()]),
        layout=dict(
            title="Cumulative Return",
            xaxis=dict(tickangle=45, rangebreaks=kwargs.get("rangebreaks", guess_plotly_rangebreaks(t_df.select(pl.col("datetime"))))),
        ),
    ).figure

    tmp_df = t_df.select(["long-short", "long-average"])
    _bin_size = ((tmp_df.max() - tmp_df.min()) / 20).min(axis=1)[0]
    group_hist_figure = SubplotsGraph(
        t_df.select(["datetime","long-short", "long-average"]),
        index_column = "datetime",
        kind_map=dict(kind="DistplotGraph", kwargs=dict(bin_size=_bin_size)),
        subplots_kwargs=dict(
            rows=1,
            cols=2,
            print_grid=False,
            subplot_titles=["long-short", "long-average"],
        ),
    ).figure

    return group_scatter_figure, group_hist_figure

def _plot_qq(data: pl.Series = None, dist=stats.norm) -> go.Figure:
    """

    :param data:
    :param dist:
    :return:
    """
    # NOTE: plotly.tools.mpl_to_plotly not actively maintained, resulting in errors in the new version of matplotlib,
    # ref: https://github.com/plotly/plotly.py/issues/2913#issuecomment-730071567
    # removing plotly.tools.mpl_to_plotly for greater compatibility with matplotlib versions
    _plt_fig = sm.qqplot(data.drop_nulls().to_numpy(), dist=dist, fit=True, line="45")
    plt.close(_plt_fig)
    qqplot_data = _plt_fig.gca().lines
    fig = go.Figure()

    fig.add_trace(
        {
            "type": "scatter",
            "x": qqplot_data[0].get_xdata(),
            "y": qqplot_data[0].get_ydata(),
            "mode": "markers",
            "marker": {"color": "#19d3f3"},
        }
    )

    fig.add_trace(
        {
            "type": "scatter",
            "x": qqplot_data[1].get_xdata(),
            "y": qqplot_data[1].get_ydata(),
            "mode": "lines",
            "line": {"color": "#636efa"},
        }
    )
    del qqplot_data
    return fig

def _pred_ic(
    pred_label: pl.DataFrame = None, 
    analysis_column = "score",
    methods: Sequence[Literal["IC", "Rank IC"]] = ("IC", "Rank IC"),
     **kwargs
) -> tuple:
    """

    :param pred_label: pl.DataFrame
    must contain one column of realized return with name `label` and one column of predicted score names `score`.
    :param methods: Sequence[Literal["IC", "Rank IC"]]
    IC series to plot.
    IC is sectional pearson correlation between label and score
    Rank IC is the spearman correlation between label and score
    For the Monthly IC, IC histogram, IC Q-Q plot.  Only the first type of IC will be plotted.
    :return:
    """
    _methods_mapping = {"IC": "pearson", "Rank IC": "spearman"}
    _ic = pred_label.groupby("datetime").agg(
                [pl.corr("label",analysis_column, method = _methods_mapping["IC"]).alias("IC")]).sort("datetime")
    ic_df = pred_label.groupby("datetime").agg(
                [pl.corr("label",analysis_column, method = _methods_mapping["Rank IC"]).alias("Rank IC")]).sort("datetime")
    ic_df = ic_df.join(_ic,on="datetime",how="inner")

    monthly_ic = _ic.with_columns([pl.col("datetime").dt.month().alias("Month"),pl.col("datetime").dt.year().alias("Year")])
    monthly_ic = monthly_ic.groupby(["Year","Month"], maintain_order=True).agg(pl.col("IC").mean())
    ic_bar_figure,rank_ic_bar_figure = ic_figure(ic_df)

    ic_heatmap_figure = HeatmapGraph(
        monthly_ic.pivot(values="IC",index="Year",columns="Month"),
        index_column = "Year",
        layout=dict(title="Monthly IC", xaxis=dict(dtick=1), yaxis=dict(tickformat="04d", dtick=1)),
        graph_kwargs=dict(xtype="array", ytype="array"),
    ).figure

    dist = stats.norm
    _qqplot_fig = _plot_qq(_ic, dist)

    if isinstance(dist, stats.norm.__class__):
        dist_name = "Normal"
    else:
        dist_name = "Unknown"


    _bin_size = ((_ic.max() - _ic.min())/20)[0,"IC"]
    _sub_graph_data = [
        (
            "IC",
            dict(
                row=1,
                col=1,
                name="",
                kind="DistplotGraph",
                graph_kwargs=dict(bin_size=_bin_size),
            ),
        ),
        (_qqplot_fig, dict(row=1, col=2)),
    ]
    ic_hist_figure = SubplotsGraph(
        _ic.drop_nulls(),
        index_column = "datetime",
        kind_map=dict(kind="HistogramGraph", kwargs=dict()),
        subplots_kwargs=dict(
            rows=1,
            cols=2,
            print_grid=False,
            subplot_titles=["IC", "IC %s Dist. Q-Q" % dist_name],
        ),
        sub_graph_data=_sub_graph_data,
        layout=dict(
            yaxis2=dict(title="Observed Quantile"),
            xaxis2=dict(title=f"{dist_name} Distribution Quantile"),
        ),
    ).figure

    return (ic_bar_figure, rank_ic_bar_figure, ic_heatmap_figure, ic_hist_figure)


def _pred_autocorr(
    pred_label: pl.DataFrame, 
    analysis_column,
    lag=1, 
    **kwargs
) -> tuple:
    pred = pred_label.clone()
    pred =  pred.with_columns([
        pl.col(analysis_column).shift(lag).over("instrument").alias(f"{analysis_column}_last")
    ])

    ac = pred.groupby("datetime").agg([pl.corr(pl.col(analysis_column).rank()/pl.count(),pl.col(f"{analysis_column}_last").rank()/pl.count())]).sort("datetime")
    ac_figure = ScatterGraph(
        ac,
        layout=dict(
            title="Auto Correlation",
            xaxis=dict(tickangle=45, rangebreaks=kwargs.get("rangebreaks", guess_plotly_rangebreaks(ac.select(pl.col("datetime"))))),
        ),
    ).figure
    return (ac_figure,)

def _pred_turnover(
    pred_label: pl.DataFrame, 
    analysis_column,
    N=5, 
    lag=1, 
    **kwargs
) -> tuple:
    pred = pred_label.clone()
    pred =  pred.with_columns([
        pl.col(analysis_column).shift(lag).over("instrument").alias(f"{analysis_column}_last")
    ])
    top = pred.groupby("datetime").agg([
        (1.0 - (pl.col("instrument").sort_by(analysis_column,descending=True).slice(0,pl.count()//N).is_in(
            pl.col("instrument").sort_by(f"{analysis_column}_last",descending=True).slice(0,pl.count()//N))
              ).sum()/(pl.count()//N)).alias("top")
    ]).sort("datetime")
    
    bottom = pred.groupby("datetime").agg([
        (1.0 - (pl.col("instrument").sort_by(analysis_column,descending=False).slice(0,pl.count()//N).is_in(
            pl.col("instrument").sort_by(f"{analysis_column}_last",descending=False).slice(0,pl.count()//N))
              ).sum()/(pl.count()//N)).alias("bottom")
    ]).sort("datetime")
    r_df = top.join(bottom,on="datetime",how="inner")
    turnover_figure_top = ScatterGraph(
        r_df.select(["datetime","top"]),
        layout=dict(
            title="Top Turnover",
            xaxis=dict(tickangle=45, rangebreaks=kwargs.get("rangebreaks", guess_plotly_rangebreaks(r_df.select(pl.col("datetime"))))),
        ),
    ).figure
    turnover_figure_bottom = ScatterGraph(
        r_df.select(["datetime","bottom"]),
        layout=dict(
            title="Bottom Turnover",
            xaxis=dict(tickangle=45, rangebreaks=kwargs.get("rangebreaks", guess_plotly_rangebreaks(r_df.select(pl.col("datetime"))))),
        ),
    ).figure
    return (turnover_figure_top, turnover_figure_bottom)


def ic_figure(ic_df: pl.DataFrame, **kwargs) -> go.Figure:
    r"""IC figure

    :param ic_df: ic DataFrame
    :param show_nature_day: whether to display the abscissa of non-trading day
    :param \*\*kwargs: contains some parameters to control plot style in plotly. Currently, supports
       - `rangebreaks`: https://plotly.com/python/time-series/#Hiding-Weekends-and-Holidays
    :return: plotly.graph_objs.Figure
    """
    ic_bar_figure = BarGraph(
        ic_df.select(["datetime", "IC"]),
        layout=dict(
            title="Information Coefficient (IC)",
            xaxis=dict(tickangle=45, rangebreaks=kwargs.get("rangebreaks", guess_plotly_rangebreaks(ic_df.select(pl.col("datetime"))))),
        ),
    ).figure
    rank_ic_bar_figure = BarGraph(
        ic_df.select(["datetime", "Rank IC"]),
        layout=dict(
            title="Ranked Information Coefficient (Rank IC)",
            xaxis=dict(tickangle=45, rangebreaks=kwargs.get("rangebreaks", guess_plotly_rangebreaks(ic_df.select(pl.col("datetime"))))),
        ),
    ).figure
    return (ic_bar_figure,rank_ic_bar_figure)

def factor_performance_graph(
    pred_label: pl.DataFrame,
    analysis_column: str = "score",
    lag: int = 1,
    N: int = 5,
    reverse=False,
    rank=False,
    graph_names: list = ["group_return", "pred_ic", "pred_autocorr","pred_turnover"],
    show_notebook: bool = True,
    show_nature_day: bool = False,
    **kwargs,
) -> [list, tuple]:
    """
    plot performance of factors.
    """
    figure_list = []
    for graph_name in graph_names:
        fun_res = eval(f"_{graph_name}")(
            pred_label=pred_label, analysis_column = analysis_column, lag=lag, N=N, reverse=reverse, rank=rank, **kwargs
        )
        figure_list += fun_res

    if show_notebook:
        BarGraph.show_graph_in_notebook(figure_list)
    else:
        return figure_list





