import datetime
from typing import Union
import pandas as pd 
import pytz

if __name__ =="__main__":
      import argparse
      parser = argparse.ArgumentParser()
      parser.add_argument('--name')
      args = parser.parse_args()
      df = pd.read_csv(
            args.name,
            header=None,
            names=["trade_id","price","quantity","quoteQty","timestamp","buyer_maker"]
            )
      df.to_csv("processed"+args.name)