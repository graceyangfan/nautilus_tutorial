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

from nautilus_trader.core.correctness cimport Condition
from nautilus_trader.model.data.bar cimport Bar
from nautilus_trader.model.data.bar cimport BarType
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
        double small_buy_value,
        double big_buy_value,
        double small_sell_value,
        double big_sell_value,
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

        self.small_buy_value = small_buy_value 
        self.big_buy_value = big_buy_value
        self.small_sell_value = small_sell_value
        self.big_sell_value = big_sell_value

    def __getstate__(self):
        return (
            *super().__getstate__(),
            str(self.small_buy_value),
            str(self.big_buy_value),
            str(self.small_sell_value),
            str(self.big_sell_value),
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
            f"small_buy_value={self.small_buy_value}, "
            f"big_buy_value={self.big_buy_value}, "
            f"small_sell_value={self.small_sell_value}, "
            f"big_sell_value={self.big_sell_value}, "
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
            small_buy_value=values["small_buy_value"],
            big_buy_value=values["big_buy_value"],
            small_sell_value=values["small_sell_value"],
            big_sell_value=values["big_sell_value"],
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
            "small_buy_value": obj.small_buy_value,
            "big_buy_value": obj.big_buy_value,
            "small_sell_value": obj.small_sell_value,
            "big_sell_value": obj.big_sell_value,
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
    def small_buy_value(self) -> double:
        """
        Return the small_buy_value.

        Returns
        -------
        double

        """
        return self.small_buy_value

    @property
    def big_buy_value(self) -> double:
        """
        Return the big_buy_value.

        Returns
        -------
        double

        """
        return self.big_buy_value

    @property
    def small_sell_value(self) -> double:
        """
        Return the small_sell_value.

        Returns
        -------
        double

        """
        return self.small_sell_value

    @property
    def big_sell_value(self) -> double:
        """
        Return the big_sell_value.

        Returns
        -------
        double

        """
        return self.big_sell_value
