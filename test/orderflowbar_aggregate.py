
import argparse
from datetime import datetime
import pandas as pd
import os 
import glob 
from tqdm import tqdm

from nautilus_trader.model.data.orderflowbar import OrderFlowBar
from nautilus_trader.model.data.aggregate_orderflow_bar import OrderFlowBarBuilder
from nautilus_trader.model.data.bar import BarSpecification
from nautilus_trader.model.data.bar import BarType
from nautilus_trader.model.enums import BarAggregation
from nautilus_trader.model.enums import PriceType
from nautilus_trader.persistence.wranglers import TradeTickDataWrangler
from nautilus_trader.common.clock import  TestClock
from nautilus_trader.persistence.catalog.parquet import ParquetDataCatalog
from nautilus_trader.serialization.base import register_serializable_object
from nautilus_trader.serialization.arrow.serializer import make_dict_serializer,make_dict_deserializer,register_arrow

def ts_parser(time_in_secs: str) -> datetime:
    """Parses timestamp string into a datetime object."""
    return datetime.utcfromtimestamp(int(time_in_secs) / 1_000.0)


def read_df(filenames):
    df = pd.DataFrame([])
    for file in tqdm(filenames):
        df_in = pd.read_csv(
            file,
            index_col="transact_time",
            date_parser=ts_parser,
            parse_dates=True,
            skiprows=1,
            names = ["agg_trade_id","price","quantity","first_trade_id","last_trade_id","transact_time","is_buyer_maker"],
            #compression='zip'
        )
        df = pd.concat([df,df_in],axis=0)
   
    df = df[["agg_trade_id","price","quantity","is_buyer_maker"]]
    #rename columns
    df.columns = ["trade_id","price","quantity","buyer_maker"]
    return df 

def aggregate_orderflow_bar(
    instrument,
    df,
    imbalance_ratio,
    bar_spec = BarSpecification(10, BarAggregation.SECOND, PriceType.LAST),
):
    wrangler = TradeTickDataWrangler(instrument=instrument)
    ticks = wrangler.process(df)
    clock = TestClock()
    start_time = ticks[0].ts_event
    clock.set_time(start_time)

    bar_type = BarType(instrument.id, bar_spec)

    handlers = []   
    def on_bar(bar):                                         
        handlers.append(bar) 

    bar_builder = OrderFlowBarBuilder(
        instrument,
        bar_type,
        on_bar,
        imbalance_ratio,
        clock
    )

    for tick in tqdm(ticks):
        bar_builder.handle_trade_tick(tick)
        events = clock.advance_time(tick.ts_event)
        for event in events:
            event.handle()

    return handlers


if __name__ == "__main__":
    
    register_serializable_object(OrderFlowBar, OrderFlowBar.to_dict, OrderFlowBar.from_dict)
    register_arrow(
        data_cls=OrderFlowBar, 
        serializer=make_dict_serializer(OrderFlowBar.schema()),
        deserializer=make_dict_deserializer(OrderFlowBar),
        schema=OrderFlowBar.schema(),
    )

    parser = argparse.ArgumentParser()
    parser.add_argument('--symbol', default='ETHUSDT')
    parser.add_argument('--ratio',default=1.2,type=float) 
    parser.add_argument("--min",default=1,type=int) 
    parser.add_argument("--venue",default='BINANCE') 
    args = parser.parse_args()
    
    catalog = ParquetDataCatalog("./catalog/.")
    instrument_id = f"{args.symbol}-PERP.{args.venue}"  
    instrument = catalog.instruments(instrument_ids=[instrument_id],as_nautilus=True)[0]

    filenames = glob.glob(f"./data/{args.symbol}*.csv")
    df = read_df(filenames)
    bars = aggregate_orderflow_bar(
        instrument,
        df,
        args.ratio,
        bar_spec = BarSpecification(args.min, BarAggregation.MINUTE, PriceType.LAST),
    )
    catalog.write_data(bars)