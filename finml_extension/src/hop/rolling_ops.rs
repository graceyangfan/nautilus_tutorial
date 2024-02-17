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
    let mut result: Vec<Series> = Vec::new();
    // Add index series
    let index_series = Series::new("index", (0..inputs[0].len() as i64).collect::<Vec<_>>());
    result.push(index_series);
    // Add x series
    let x_series = inputs[0].clone().with_name("x");
    result.push(x_series);

    // Add series from inputs[1] to last one with names from by_columns[0] to last one
    if let Some(by_columns) = &kwargs.by_columns {
        for (i, series) in inputs[1..].iter().enumerate() {
            let name = by_columns[i].clone();
            result.push(series.clone().with_name(&name));
        }
    }
    let df = DataFrame::new(result)?;
    let shift_window = kwargs.window as i32 - 1; 
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
                    period: Duration::parse(&format!("{}i", kwargs.window)),
                    offset: Duration::parse(&format!("-{}i", kwargs.window)),
                    closed_window: ClosedWindow::Right,
                    ..Default::default()
                }
            )
            .agg([
                col("x").arg_max().alias(out_name)
            ])
            .with_columns([
                match &kwargs.by_columns {
                    Some(by_columns) => col(out_name)
                                        .shift(lit(-shift_window))
                                        .shift(lit(shift_window))
                                        .over(by_columns.iter().map(|name| col(name)).collect::<Vec<_>>()),
                   None =>  col(out_name)
                            .shift(lit(-shift_window))
                            .shift(lit(shift_window)),
                }
            ])
            .collect()?;
    out.drop_in_place(out_name)
}



