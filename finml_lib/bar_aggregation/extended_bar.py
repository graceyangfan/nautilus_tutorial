# -------------------------------------------------------------------------------------------------
#  Copyright (C) 2015-2022 Nautech Systems Pty Ltd. All rights reserved.
#  https://nautechsystems.io
#
#  Licensed under the GNU Lesser General Public License Version 3.0 (the "License");
#  You may not use this file except in compliance with the License.
#  You may obtain a copy of the License at https://www.gnu.org/licenses/lgpl-3.0.en.html
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# -------------------------------------------------------------------------------------------------

from decimal import Decimal
from typing import Any, Optional

from nautilus_trader.model.data.bar import Bar
from nautilus_trader.model.data.bar import BarType
from nautilus_trader.model.data.ticker import Ticker
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.objects import Price
from nautilus_trader.model.objects import Quantity
import pyarrow as pa
from nautilus_trader.serialization.base import register_serializable_object
from nautilus_trader.serialization.arrow.serializer import register_parquet

class ExtendedBar(Bar):
    """
    Represents an aggregated `Extended` bar.

    This data type includes the raw data provided by `Extended`.

    Parameters
    ----------
    bar_type : BarType
        The bar type for this bar.
    open : Price
        The bars open price.
    high : Price
        The bars high price.
    low : Price
        The bars low price.
    close : Price
        The bars close price.
    volume : Quantity
        The bars volume.
    bids_volume: List 
        Different quantity level of buy_makers.
    asks_volume: List 
        Different quantity level of sell_makers.
    ts_event : uint64_t
        The UNIX timestamp (nanoseconds) when the data event occurred.
    ts_init : uint64_t
        The UNIX timestamp (nanoseconds) when the data object was initialized.

    """

    def __init__(
        self,
        bar_type: BarType,
        open: Price,
        high: Price,
        low: Price,
        close: Price,
        volume: Quantity,
        bids_value_level_0: float,
        bids_value_level_1: float,
        bids_value_level_2: float,
        bids_value_level_3: float,
        bids_value_level_4: float,
        asks_value_level_0: float,
        asks_value_level_1: float,
        asks_value_level_2: float,
        asks_value_level_3: float,
        asks_value_level_4: float,
        ts_event: int,
        ts_init: int,
    ):
        super().__init__(
            bar_type=bar_type,
            open=open,
            high=high,
            low=low,
            close=close,
            volume=volume,
            ts_event=ts_event,
            ts_init=ts_init,
        )

        self.bids_value_level_0 = bids_value_level_0
        self.bids_value_level_1 = bids_value_level_1
        self.bids_value_level_2 = bids_value_level_2
        self.bids_value_level_3 = bids_value_level_3
        self.bids_value_level_4 = bids_value_level_4
        self.asks_value_level_0 = asks_value_level_0
        self.asks_value_level_1 = asks_value_level_1
        self.asks_value_level_2 = asks_value_level_2
        self.asks_value_level_3 = asks_value_level_3
        self.asks_value_level_4 = asks_value_level_4

    def __getstate__(self):
        return (
            *super().__getstate__(),
            str(self.bids_value_level_0),
            str(self.bids_value_level_1),
            str(self.bids_value_level_2),
            str(self.bids_value_level_3),
            str(self.bids_value_level_4),
            str(self.asks_value_level_0),
            str(self.asks_value_level_1),
            str(self.asks_value_level_2),
            str(self.asks_value_level_3),
            str(self.asks_value_level_4),
        )

    def __setstate__(self, state):

        super().__setstate__(state[:15])

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"bar_type={self.bar_type}, "
            f"open={self.open}, "
            f"high={self.high}, "
            f"low={self.low}, "
            f"close={self.close}, "
            f"volume={self.volume}, "
            f"bids_value_level_0={self.bids_value_level_0}, "
            f"bids_value_level_1={self.bids_value_level_1}, "
            f"bids_value_level_2={self.bids_value_level_2}, "
            f"bids_value_level_3={self.bids_value_level_3}, "
            f"bids_value_level_4={self.bids_value_level_4}, "
            f"asks_value_level_0={self.asks_value_level_0}, "
            f"asks_value_level_1={self.asks_value_level_1}, "
            f"asks_value_level_2={self.asks_value_level_2}, "
            f"asks_value_level_3={self.asks_value_level_3}, "
            f"asks_value_level_4={self.asks_value_level_4}, "
            f"ts_event={self.ts_event}, "
            f"ts_init={self.ts_init})"
        )

    @staticmethod
    def from_dict(values: dict[str, Any]) -> "ExtendedBar":
        """
        Return a `Extended` bar parsed from the given values.

        Parameters
        ----------
        values : dict[str, Any]
            The values for initialization.

        Returns
        -------
        ExtendedBar

        """
        return ExtendedBar(
            bar_type=BarType.from_str(values["bar_type"]),
            open=Price.from_str(values["open"]),
            high=Price.from_str(values["high"]),
            low=Price.from_str(values["low"]),
            close=Price.from_str(values["close"]),
            volume=Quantity.from_str(values["volume"]),
            bids_value_level_0=values["bids_value_level_0"],
            bids_value_level_1=values["bids_value_level_1"],
            bids_value_level_2=values["bids_value_level_2"],
            bids_value_level_3=values["bids_value_level_3"],
            bids_value_level_4=values["bids_value_level_4"],
            asks_value_level_0=values["asks_value_level_0"],
            asks_value_level_1=values["asks_value_level_1"],
            asks_value_level_2=values["asks_value_level_2"],
            asks_value_level_3=values["asks_value_level_3"],
            asks_value_level_4=values["asks_value_level_4"],
            ts_event=values["ts_event"],
            ts_init=values["ts_init"],
        )

    @staticmethod
    def to_dict(obj: "ExtendedBar") -> dict[str, Any]:
        """
        Return a dictionary representation of this object.

        Returns
        -------
        dict[str, Any]

        """
        return {
            "type": type(obj).__name__,
            "bar_type": str(obj.bar_type),
            "open": str(obj.open),
            "high": str(obj.high),
            "low": str(obj.low),
            "close": str(obj.close),
            "volume": str(obj.volume),
            "bids_value_level_0": obj.bids_value_level_0,
            "bids_value_level_1": obj.bids_value_level_1,
            "bids_value_level_2": obj.bids_value_level_2,
            "bids_value_level_3": obj.bids_value_level_3,
            "bids_value_level_4": obj.bids_value_level_4,
            "asks_value_level_0": obj.asks_value_level_0,
            "asks_value_level_1": obj.asks_value_level_1,
            "asks_value_level_2": obj.asks_value_level_2,
            "asks_value_level_3": obj.asks_value_level_3,
            "asks_value_level_4": obj.asks_value_level_4,
            "ts_event": obj.ts_event,
            "ts_init": obj.ts_init,
        }



ExtendedBar_SCHEMA  = pa.schema(
        {
            "bar_type": pa.dictionary(pa.int8(), pa.string()),
            #"instrument_id": pa.dictionary(pa.int64(), pa.string()),
            "open": pa.string(),
            "high": pa.string(),
            "low": pa.string(),
            "close": pa.string(),
            "volume": pa.string(),
            "bids_value_level_0": pa.float64(),
            "bids_value_level_1": pa.float64(),
            "bids_value_level_2": pa.float64(),
            "bids_value_level_3": pa.float64(),
            "bids_value_level_4": pa.float64(),
            "asks_value_level_0": pa.float64(),
            "asks_value_level_1": pa.float64(),
            "asks_value_level_2": pa.float64(),
            "asks_value_level_3": pa.float64(),
            "asks_value_level_4": pa.float64(),
            "ts_event": pa.uint64(),
            "ts_init": pa.uint64(),
        },
    metadata={"type": "ExtendedBar"},
)

register_serializable_object(ExtendedBar, ExtendedBar.to_dict, ExtendedBar.from_dict)
register_parquet(
    cls=ExtendedBar, 
    serializer=ExtendedBar.to_dict,
    deserializer= ExtendedBar.from_dict, 
    schema=ExtendedBar_SCHEMA
)