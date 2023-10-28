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
from nautilus_trader.model.data.bar cimport Bar
from nautilus_trader.model.data.bar cimport BarType
from nautilus_trader.model.objects cimport Price
from nautilus_trader.model.objects cimport Quantity

cdef class OrderFlowBar(Bar):
    def __init__(
            self,
            BarType bar_type not None,
            Price open not None,
            Price high not None,
            Price low not None,
            Price close not None,
            Quantity volume not None,
            int pressure_levels,
            int support_levels,
            double bottom_imbalance,
            double bottom_imbalance_price,
            double middle_imbalance,
            double middle_imbalance_price,
            double top_imbalance,
            double top_imbalance_price,
            double point_of_control,
            double poc_imbalance,
            double delta,
            double value_delta,
            bint up_bar,
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

            self.pressure_levels = pressure_levels
            self.support_levels = support_levels
            self.bottom_imbalance = bottom_imbalance
            self.bottom_imbalance_price  = bottom_imbalance_price 
            self.middle_imbalance = middle_imbalance 
            self.middle_imbalance_price = middle_imbalance_price 
            self.top_imbalance = top_imbalance 
            self.top_imbalance_price = top_imbalance_price
            self.point_of_control = point_of_control 
            self.poc_imbalance = poc_imbalance
            self.delta = delta 
            self.value_delta = value_delta
            self.up_bar = up_bar 
            self.tag = tag 

    def __getstate__(self):
        return (
            *super().__getstate__(),
            str(self.pressure_levels),
            str(self.support_levels),
            str(self.bottom_imbalance),
            str(self.bottom_imbalance_price),
            str(self.middle_imbalance),
            str(self.middle_imbalance_price),
            str(self.top_imbalance),
            str(self.top_imbalance_price),
            str(self.point_of_control),
            str(self.poc_imbalance),
            str(self.delta),
            str(self.value_delta),
            str(self.up_bar),
            str(self.tag)
        )

    def __setstate__(self, state):

        super().__setstate__(state[:17])

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"bar_type={self.bar_type}, "
            f"open={self.open}, "
            f"high={self.high}, "
            f"low={self.low}, "
            f"close={self.close}, "
            f"volume={self.volume}, "
            f"pressure_levels={self.pressure_levels}, "
            f"support_levels={self.support_levels}, "
            f"bottom_imbalance={self.bottom_imbalance}, "
            f"bottom_imbalance_price={self.bottom_imbalance_price}, "
            f"middle_imbalance={self.middle_imbalance}, "
            f"middle_imbalance_price={self.middle_imbalance_price}, "
            f"top_imbalance={self.top_imbalance}, "
            f"top_imbalance_price={self.top_imbalance_price}, "
            f"point_of_control={self.point_of_control}, "
            f"poc_imbalance={self.poc_imbalance}, "
            f"delta={self.delta}, "
            f"delta={self.value_delta}, "
            f"up_bar={self.up_bar}, "
            f"tag={self.tag}, "
            f"ts_event={self.ts_event}, "
            f"ts_init={self.ts_init})"
        )

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
                "pressure_levels": pa.uint64(),
                "support_levels": pa.uint64(),
                "bottom_imbalance": pa.float64(),
                "bottom_imbalance_price": pa.float64(),
                "middle_imbalance": pa.float64(),
                "middle_imbalance_price": pa.float64(),
                "top_imbalance": pa.float64(),
                "top_imbalance_price": pa.float64(),
                "point_of_control": pa.float64(),
                "poc_imbalance": pa.float64(),
                "delta":pa.float64(),
                "value_delta":pa.float64(),
                "up_bar":pa.bool_(),
                "tag":pa.uint64(),
                "ts_event": pa.uint64(),
                "ts_init": pa.uint64(),
            },
            metadata={"type": "OrderFlowBar"},
        )


    @staticmethod
    cdef OrderFlowBar from_dict_c(dict values):
        """
        Return a `OrderFlowBar` bar parsed from the given values.

        Parameters
        ----------
        values : dict[str, Any]
            The values for initialization.

        Returns
        -------
        OrderFlowBar

        """
        Condition.not_none(values, "values")

        return OrderFlowBar(
            bar_type=BarType.from_str(values["bar_type"]),
            open=Price.from_str(values["open"]),
            high=Price.from_str(values["high"]),
            low=Price.from_str(values["low"]),
            close=Price.from_str(values["close"]),
            volume=Quantity.from_str(values["volume"]),
            pressure_levels=values["pressure_levels"],
            support_levels=values["support_levels"],
            bottom_imbalance=values["bottom_imbalance"],
            bottom_imbalance_price=values["bottom_imbalance_price"],
            middle_imbalance=values["middle_imbalance"],
            middle_imbalance_price=values["middle_imbalance_price"],
            top_imbalance=values["top_imbalance"],
            top_imbalance_price=values["top_imbalance_price"],
            point_of_control=values["point_of_control"], 
            poc_imbalance=values["poc_imbalance"],
            delta=values["delta"],
            value_delta=values["value_delta"],
            up_bar=values["up_bar"],
            tag=values["tag"],
            ts_event=values["ts_event"],
            ts_init=values["ts_init"],
        )

    @staticmethod
    cdef dict to_dict_c(OrderFlowBar obj):
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
            "pressure_levels": obj.pressure_levels,
            "support_levels": obj.support_levels,
            "bottom_imbalance": obj.bottom_imbalance,
            "bottom_imbalance_price": obj.bottom_imbalance_price,
            "middle_imbalance": obj.middle_imbalance,
            "middle_imbalance_price": obj.middle_imbalance_price,
            "top_imbalance": obj.top_imbalance,
            "top_imbalance_price": obj.top_imbalance_price,
            "point_of_control": obj.point_of_control,
            "poc_imbalance": obj.poc_imbalance,
            "delta":obj.delta,
            "value_delta":obj.value_delta,
            "up_bar":obj.up_bar,
            "tag":obj.tag,
            "ts_event": obj.ts_event,
            "ts_init": obj.ts_event,
        }

    @staticmethod
    def from_dict(dict values) -> OrderFlowBar:
        """
        Return a bar parsed from the given values.

        Parameters
        ----------
        values : dict[str, object]
            The values for initialization.

        Returns
        -------
        OrderFlowBar

        """
        return OrderFlowBar.from_dict_c(values)

    @staticmethod
    def to_dict(OrderFlowBar obj):
        """
        Return a dictionary representation of this object.

        Returns
        -------
        dict[str, object]

        """
        return OrderFlowBar.to_dict_c(obj)

    @property
    def pressure_levels(self) -> int:
        """
        Return the pressure_levels.

        Returns
        -------
        int

        """
        return self.pressure_levels

    @property
    def support_levels(self) -> int:
        """
        Return the support_levels.

        Returns
        -------
        int

        """
        return self.support_levels

    @property
    def bottom_imbalance(self) -> double:
        """
        Return the bottom_imbalance.

        Returns
        -------
        double

        """
        return self.bottom_imbalance

    @property
    def bottom_imbalance_price (self) -> double:
        """
        Return the bottom_imbalance_price .

        Returns
        -------
        double

        """
        return self.bottom_imbalance_price 

    @property
    def middle_imbalance(self) -> double:
        """
        Return the middle_imbalance.

        Returns
        -------
        double

        """
        return self.middle_imbalance

    @property
    def middle_imbalance_price(self) -> double:
        """
        Return the middle_imbalance_price.

        Returns
        -------
        double

        """
        return self.middle_imbalance_price

    @property
    def top_imbalance(self) -> double:
        """
        Return the top_imbalance.

        Returns
        -------
        double

        """
        return self.top_imbalance

    @property
    def top_imbalance_price(self) -> double:
        """
        Return the top_imbalance_price.

        Returns
        -------
        double

        """
        return self.top_imbalance_price

    @property
    def point_of_control(self) -> double:
        """
        Return the point_of_control.

        Returns
        -------
        double

        """
        return self.point_of_control

    @property
    def poc_imbalance(self) -> double:
        """
        Return the poc_imbalance.

        Returns
        -------
        double

        """
        return self.poc_imbalance

    @property
    def delta(self) -> double:
        """
        Return the delta.

        Returns
        -------
        double

        """
        return self.delta

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
    def up_bar(self) -> bool:
        """
        Return the up_bar.

        Returns
        -------
        bool

        """
        return self.up_bar


    @property
    def tag(self) -> int:
        """
        Return the tag.

        Returns
        -------
        int

        """
        return self.tag

