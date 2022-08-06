import os
import shutil
from functools import partial
from pathlib import Path
from decimal import Decimal
import pandas as pd
from nautilus_trader.core.datetime import dt_to_unix_nanos
from nautilus_trader.model.c_enums.bar_aggregation import BarAggregation
from nautilus_trader.model.data.bar import Bar, BarType, BarSpecification
from nautilus_trader.model.enums import AggregationSource, PriceType
from nautilus_trader.model.instruments.currency_pair import CurrencyPair
from nautilus_trader.model.identifiers import InstrumentId, Symbol, Venue
from nautilus_trader.model.objects import Money
from nautilus_trader.model.currencies import USDT
from nautilus_trader.model.objects import Price, Quantity
from nautilus_trader.persistence.catalog.parquet import ParquetDataCatalog
from nautilus_trader.persistence.external.core import process_files, write_objects
from nautilus_trader.persistence.external.readers import ParquetReader
from nautilus_trader.core.datetime import secs_to_nanos

import os, shutil
# fs = fsspec.filesystem("file")
# raw_files = fs.glob(data_file)
# assert raw_files, f"Unable to find file: {data_file}"

from nautilus_trader.model.currencies import ETH

def parser(data, instrument):
    """Parser function for stock OHLC data, for use with CSV Reader"""
    dt = data["open_time"].astype(int)*1000000
    
    bar_type = BarType(
        instrument.id,
        BarSpecification(1, BarAggregation.MINUTE, PriceType.LAST),
        AggregationSource.EXTERNAL,  # Bars are aggregated in the data already
    )
    
    def generate_bar(data,ts_events,ts_inits):
        return Bar(
        bar_type=bar_type,
        open=Price.from_str(str(data[0])),
        high=Price.from_str(str(data[1])),
        low=Price.from_str(str(data[2])),
        close=Price.from_str(str(data[3])),
        volume=Quantity.from_str(str(data[4])),
        ts_event=ts_events,
        ts_init=ts_inits,
        check=True,
    )
    select_columns=["open","high","low","close","volume"]
    bars = list(map(generate_bar,data[select_columns].values,dt.values,dt.values))
    yield from bars



def get_ETH_USDT_instrument(exchange):
    return CurrencyPair(
        instrument_id=InstrumentId(
            symbol=Symbol("ETHUSDT"),
            venue=Venue(exchange),
        ),
        native_symbol=Symbol("ETHUSDT"),
        base_currency=ETH,
        quote_currency=USDT,
        price_precision=2,
        size_precision=3,
        price_increment=Price(1e-02, precision=2),
        size_increment=Quantity(1e-3, precision=3),
        lot_size=None,
        max_quantity=Quantity(9000, precision=3),
        min_quantity=Quantity(1e-03, precision=3),
        max_notional=None,
        min_notional=Money(1.00, USDT),
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
    CATALOG_PATH = os.getcwd() + "/catalog"
    catalog = ParquetDataCatalog(CATALOG_PATH)
    ETH = get_ETH_USDT_instrument("BINANCE")
    print(ETH)
    process_files(
        glob_path=os.path.join("./catalog/compressed","ETH-USDT.parquet"),
        reader=ParquetReader(partial(parser, instrument=ETH)),
        catalog=catalog,
    )

    write_objects(catalog, [ETH])


    assert catalog.instruments(as_nautilus=True)
    assert catalog.bars(as_nautilus=True)
