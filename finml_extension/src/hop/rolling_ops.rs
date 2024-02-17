use polars::prelude::*;
use polars_core::series::IsSorted;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

#[derive(Deserialize, Debug)]
pub(crate) struct GroupByRollingKwargs {
    pub(crate) window: u32,
    pub(crate) by_columns: Option<Vec<String>>,
}


#[polars_expr(output_type=Float64)]
pub fn pl_rolling_idxmax(inputs: &[Series], kwargs: GroupByRollingKwargs) ->PolarsResult<Series> 
{
    let out_name = "idxmax";
    let index_series: Vec<_> = (0..inputs[0].len() as i64).collect();
    let df = df!(
        "x" => inputs[0].clone(),
        "index" => index_series
    )?;
    //use rolling 
    let mut out = df
            .lazy()
            .with_column(col("index").set_sorted_flag(IsSorted::Ascending))
            .group_by_rolling(
                col("index"),
                //vec for kwargs.by_columns 
                match &kwargs.by_columns {
                    Some(by_columns) => by_columns.iter().map(|name| col(name)).collect(),
                    None => vec![],
                },
                RollingGroupOptions {
                    period: Duration::parse(&format!("{}i", &kwargs.window)),
                    offset: Duration::parse(&format!("-{}i", &kwargs.window)),
                    closed_window: ClosedWindow::Right,
                    ..Default::default()
                }
            )
            .agg([col("x").arg_max().alias(out_name)])
            .select([when(col("index").lt(lit(kwargs.window))).then(lit(NULL)).otherwise(col(out_name)).alias(out_name)])
            .collect()?;
    out.drop_in_place(out_name)
}



