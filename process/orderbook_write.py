import os
import pandas as pd
from nautilus_trader.persistence.catalog.parquet import ParquetDataCatalog
from nautilus_trader.persistence.external.core import process_files, write_objects
from nautilus_trader.persistence.external.readers import CSVReader
from nautilus_trader.model.enums import BookTypeParser
from nautilus_trader.model.orderbook.data import OrderBookSnapshot
if __name__ == "__main__":
    import argparse
    import glob 
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbol',default='MATICUSDT')
    parser.add_argument("--venue",default='BINANCE') 
    parser.add_argument("--fileloction",default="data/compressed") 
    args = parser.parse_args()
    CATALOG_PATH ="catalog"
    catalog = ParquetDataCatalog(CATALOG_PATH)
    instrument_id = f"{args.symbol}-PERP.{args.venue}"
    instrument = catalog.instruments(instrument_ids=[instrument_id],as_nautilus=True)[0]

    fileloc = args.fileloction
    input_files = glob.glob("data/compressed/"+args.symbol+"-depth5*")
    print(input_files)

    book_type  = BookTypeParser.from_str_py(
        "L2_MBP"
    )
    def parser(df):
        for idx,r in df.iterrows():
            ts = int(r['timestamp']/1000*10**9) #ms => ns 
            book =   OrderBookSnapshot(
                        instrument_id=instrument.id,
                        book_type= book_type,
                        bids=[
                            (r['bp'+str(i)],r['bz'+str(i)]) for i in range(5,0,-1)
                        ],
                        asks=[
                            (r['ap'+str(i)],r['az'+str(i)]) for i in range(1,6)
                        ],
                        ts_event= ts,
                        ts_init= ts,
                    )
            yield book 

    process_files(
        glob_path=input_files,
        reader=CSVReader(
            block_parser= parser, 
        ),
        catalog=catalog,
    )

    write_objects(catalog, [instrument])

    
    
    
    
    
    
