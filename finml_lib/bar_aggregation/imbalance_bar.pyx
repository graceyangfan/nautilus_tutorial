# -------------------------------------------------------------------------------------------------
#  Copyright (C) 2015-2023 Nautech Systems Pty Ltd. All rights reserved.
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

from libc.stdint cimport uint64_t
import pyarrow as pa

from nautilus_trader.core.correctness cimport Condition
from nautilus_trader.model.data cimport Bar
from nautilus_trader.model.data cimport BarType
from nautilus_trader.model.objects cimport Price
from nautilus_trader.model.objects cimport Quantity

cdef class ImbalanceBar(Bar):
    """
    Represents an aggregated `Imbalance` bar.

    This data type includes the raw data provided by `Imbalance`.

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
    ts_event : uint64_t
        The UNIX timestamp (nanoseconds) when the data event occurred.
    ts_init : uint64_t
        The UNIX timestamp (nanoseconds) when the data object was initialized.
    """

    def __init__(
        self,
        BarType bar_type not None,
        Price open not None,
        Price high not None,
        Price low not None,
        Price close not None,
        Quantity volume not None,
        double big_buy_ratio,
        double big_net_buy_ratio,
        double big_buy_power,
        double big_net_buy_power,
        double value_delta,
        int tag,
        uint64_t ts_event,
        uint64_t ts_init,
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

        self.big_buy_ratio = big_buy_ratio 
        self.big_net_buy_ratio = big_net_buy_ratio
        self.big_buy_power = big_buy_power
        self.big_net_buy_power = big_net_buy_power
        self.value_delta = value_delta 
        self.tag = tag 

    def __getstate__(self):
        return (
            *super().__getstate__(),
            str(self.big_buy_ratio),
            str(self.big_net_buy_ratio),
            str(self.big_buy_power),
            str(self.big_net_buy_power),
            str(self.value_delta),
            str(self.tag),
        )

    def __setstate__(self, state):

        super().__setstate__(state[:15])

    @classmethod
    def schema(cls):
        return pa.schema(
            {
                "bar_type": pa.dictionary(pa.int8(), pa.string()),
                "open": pa.string(),
                "high": pa.string(),
                "low": pa.string(),
                "close": pa.string(),
                "volume": pa.string(),
                "big_buy_ratio": pa.float64(),
                "big_net_buy_ratio": pa.float64(),
                "big_buy_power": pa.float64(),
                "big_net_buy_power": pa.float64(),
                "value_delta": pa.float64(),
                "tag":pa.uint64(),
                "ts_event": pa.uint64(),
                "ts_init": pa.uint64(),
            },
            metadata={"type": "ImbalanceBar"},
        )

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"bar_type={self.bar_type}, "
            f"open={self.open}, "
            f"high={self.high}, "
            f"low={self.low}, "
            f"close={self.close}, "
            f"volume={self.volume}, "
            f"big_buy_ratio={self.big_buy_ratio}, "
            f"big_net_buy_ratio={self.big_net_buy_ratio}, "
            f"big_buy_power={self.big_buy_power}, "
            f"big_net_buy_power={self.big_net_buy_power}, "
            f"value_delta={self.value_delta}, "
            f"tag={self.tag}, "
            f"ts_event={self.ts_event}, "
            f"ts_init={self.ts_init})"
        )

    @staticmethod
    cdef ImbalanceBar from_dict_c(dict values):
        """
        Return a `Imbalance` bar parsed from the given values.

        Parameters
        ----------
        values : dict[str, Any]
            The values for initialization.

        Returns
        -------
        ImbalanceBar

        """
        Condition.not_none(values, "values")

        return ImbalanceBar(
            bar_type=BarType.from_str(values["bar_type"]),
            open=Price.from_str(values["open"]),
            high=Price.from_str(values["high"]),
            low=Price.from_str(values["low"]),
            close=Price.from_str(values["close"]),
            volume=Quantity.from_str(values["volume"]),
            big_buy_ratio=values["big_buy_ratio"],
            big_net_buy_ratio=values["big_net_buy_ratio"],
            big_buy_power=values["big_buy_power"],
            big_net_buy_power=values["big_net_buy_power"],
            value_delta=values["value_delta"],
            tag=values["tag"],
            ts_event=values["ts_event"],
            ts_init=values["ts_init"],
        )

    @staticmethod
    cdef dict to_dict_c(Bar obj):
        """
        Return a dictionary representation of this object.

        Returns
        -------
        dict[str, Any]

        """
        Condition.not_none(obj, "obj")
        return {
            "type": type(obj).__name__,
            "bar_type": str(obj.bar_type),
            "open": str(obj.open),
            "high": str(obj.high),
            "low": str(obj.low),
            "close": str(obj.close),
            "volume": str(obj.volume),
            "big_buy_ratio": obj.big_buy_ratio,
            "big_net_buy_ratio": obj.big_net_buy_ratio,
            "big_buy_power": obj.big_buy_power,
            "big_net_buy_power": obj.big_net_buy_power,
            "value_delta":obj.value_delta,
            "tag":obj.tag,
            "ts_event": obj.ts_event,
            "ts_init": obj.ts_event,
        }

    @staticmethod
    def from_dict(dict values) -> ImbalanceBar:
        """
        Return a bar parsed from the given values.

        Parameters
        ----------
        values : dict[str, object]
            The values for initialization.

        Returns
        -------
        Bar

        """
        return ImbalanceBar.from_dict_c(values)

    @staticmethod
    def to_dict(ImbalanceBar obj):
        """
        Return a dictionary representation of this object.

        Returns
        -------
        dict[str, object]

        """
        return ImbalanceBar.to_dict_c(obj)

    @property
    def big_buy_ratio(self) -> double:
        """
        Return the big_buy_ratio.

        Returns
        -------
        double

        """
        return self.big_buy_ratio

    @property
    def big_net_buy_ratio(self) -> double:
        """
        Return the big_net_buy_ratio.

        Returns
        -------
        double

        """
        return self.big_net_buy_ratio

    @property
    def big_buy_power(self) -> double:
        """
        Return the big_buy_power.

        Returns
        -------
        double

        """
        return self.big_buy_power

    @property
    def big_net_buy_power(self) -> double:
        """
        Return the big_net_buy_power.

        Returns
        -------
        double

        """
        return self.big_net_buy_power


    @property
    def value_delta(self) -> double:
        """
        Return the value_delta.

        Returns
        -------
        double

        """
        return self.value_delta
    @property
    def tag(self) -> int:
        """
        Return the tag.

        Returns
        -------
        int

        """
        return self.tag