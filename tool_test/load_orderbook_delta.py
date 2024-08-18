
from nautilus_trader.persistence.catalog.parquet import ParquetDataCatalog
from nautilus_trader.persistence.loaders import BinanceOrderBookDeltaDataLoader
from nautilus_trader.persistence.wranglers_v2 import OrderBookDeltaDataWranglerV2 
import pyarrow.ipc as ipc
import pyarrow as pa

if __name__ == "__main__":
    catalog = ParquetDataCatalog("../catalog/.")
    instrument_id = "FTMUSDT-PERP.BINANCE"
    instrument = catalog.instruments(instrument_ids=[instrument_id],as_nautilus=True)[0]
    path_update = "FTMUSDT-PERP.BINANCE.feather"
    wrangler = OrderBookDeltaDataWranglerV2(instrument_id,instrument.price_precision,instrument.size_precision)

    with open(path_update, 'rb') as source:
        reader = ipc.open_stream(source)
        table = reader.read_all()   
        deltas = wrangler.from_arrow(table)
    catalog.write_data(deltas) 
    