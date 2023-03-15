import os 
import glob 
from decimal import Decimal
import argparse
import polars as pl 
from tqdm import tqdm
from nautilus_trader.persistence.catalog.parquet import ParquetDataCatalog
from nautilus_trader.backtest.data.wranglers import TradeTickDataWrangler
from nautilus_trader.data.aggregation import TimeBarAggregator
from nautilus_trader.model.data.bar import Bar
from nautilus_trader.model.data.bar import BarSpecification, BarType
from nautilus_trader.model.enums import AggressorSide, BarAggregation, PriceType
from nautilus_trader.persistence.external.core import write_objects
from nautilus_trader.model.instruments.currency_pair import CurrencyPair
from nautilus_trader.model.identifiers import Symbol
from nautilus_trader.model.identifiers import Venue
from nautilus_trader.model.currencies import BUSD,USDT 
from nautilus_trader.model.currencies import ETH
from nautilus_trader.model.objects import Money
from nautilus_trader.model.objects import Price, Quantity
from nautilus_trader.model.identifiers import InstrumentId
from finml.sampling.aggregate_bars import agg_timebar 

def get_ETH_BUSD_instrument(exchange):
    return CurrencyPair(
        instrument_id=InstrumentId(
            symbol=Symbol("ETHBUSD-PERP"),
            venue=Venue(exchange),
        ),
        native_symbol=Symbol("ETHBUSD-PERP"),
        base_currency=ETH,
        quote_currency=BUSD,
        price_precision=2,
        size_precision=3,
        price_increment=Price(1e-02, precision=2),
        size_increment=Quantity(1e-3, precision=3),
        lot_size=None,
        max_quantity=Quantity(9000, precision=3),
        min_quantity=Quantity(1e-03, precision=3),
        max_notional=None,
        min_notional=Money(1.00, BUSD),
        max_price=Price(1000000, precision=2),
        min_price=Price(0.01, precision=2),
        margin_init=Decimal("0"),
        margin_maint=Decimal("0"),
        maker_fee=Decimal("0.0004"),
        taker_fee=Decimal("0.0004"),
        ts_event=0,
        ts_init=0,
        )



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', default="../../example_data/ETHBUSD-trades-2023-02-04*.csv")
    parser.add_argument('--symbol',default='APEBUSD')
    parser.add_argument("--venue",default='BINANCE') 
    parser.add_argument('--period', type=int, default=10)
    args = parser.parse_args()

    catalog = ParquetDataCatalog("../../catalog/.")

    instrument_id = f"{args.symbol}-PERP.{args.venue}"
    # Get the instrument for the given symbol
    #instrument = asyncio.run(get_instrument(instrument_id , api_key, api_secret, args.account_type))
    if args.symbol == 'ETHBUSD':
        instrument = get_ETH_BUSD_instrument(args.venue)
    else:
        instrument = catalog.instruments(instrument_ids=[instrument_id],as_nautilus=True)[0]


    ## read data 
    wrangler = TradeTickDataWrangler(instrument=instrument)
    if os.path.exists(os.path.join("/tmp",instrument_id+".parquet")):
        df = pl.read_parquet(os.path.join("/tmp",instrument_id+".parquet"))
    else:
        df = pl.DataFrame({})
        filenames = sorted(glob.glob(args.filename))
        for file in tqdm(filenames):
            df_in = pl.read_csv(file)
            df_in.columns = ["trade_id","price","quantity","quoteQty","timestamp","buyer_maker"]

            df = pl.concat([df,df_in], how="vertical")

    # aggregate bars 
    newbars = agg_timebar(df,args.period)
    bar_spec = BarSpecification(args.period, BarAggregation.SECOND, PriceType.LAST)
    bar_type = BarType(instrument.id, bar_spec)
    handlers = [] 
    for i in range(newbars.shape[0]):
        bar = Bar(
            bar_type = bar_type,
            open = Price(newbars[i,"open"],instrument.price_precision),
            high = Price(newbars[i,"high"],instrument.price_precision),
            low = Price(newbars[i,"low"],instrument.price_precision),
            close = Price(newbars[i,"close"],instrument.price_precision),
            volume = Quantity(newbars[i,"volume"],instrument.size_precision),
            ts_event = newbars[i,"ts_event"]*10**6,
            ts_init = newbars[i,"ts_event"]*10**6,
        )
        handlers.append(bar) 
    write_objects(catalog, chunk = handlers)
    write_objects(catalog, chunk = [instrument])