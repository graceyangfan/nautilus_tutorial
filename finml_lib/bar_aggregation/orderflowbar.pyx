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
            double imbalance_pressure_price,
            double imbalance_support_price,
            int pressure_levels,
            int support_levels,
            double point_of_control,
            double top_imbalance_level1,
            double top_imbalance_level2,
            double bottom_imbalance_level1,
            double bottom_imbalance_level2,
            double delta,
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

            self.imbalance_pressure_price = imbalance_pressure_price
            self.imbalance_support_price = imbalance_support_price
            self.pressure_levels = pressure_levels 
            self.support_levels = support_levels 
            self.point_of_control = point_of_control 
            self.top_imbalance_level1 = top_imbalance_level1
            self.top_imbalance_level2 = top_imbalance_level2 
            self.bottom_imbalance_level1 = bottom_imbalance_level1
            self.bottom_imbalance_level2  = bottom_imbalance_level2 
            self.delta = delta 

    def __getstate__(self):
        return (
            *super().__getstate__(),
            str(self.imbalance_pressure_price),
            str(self.imbalance_support_price),
            str(self.pressure_levels),
            str(self.support_levels),
            str(self.point_of_control),
            str(self.top_imbalance_level1),
            str(self.top_imbalance_level2),
            str(self.bottom_imbalance_level1),
            str(self.bottom_imbalance_level2),
            str(self.delta)
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
            f"imbalance_pressure_price={self.imbalance_pressure_price}, "
            f"imbalance_support_price={self.imbalance_support_price}, "
            f"pressure_levels={self.pressure_levels}, "
            f"support_levels={self.support_levels}, "
            f"point_of_control={self.point_of_control}, "
            f"top_imbalance_level1={self.top_imbalance_level1}, "
            f"top_imbalance_level2={self.top_imbalance_level2}, "
            f"bottom_imbalance_level1={self.bottom_imbalance_level1}, "
            f"bottom_imbalance_level2={self.bottom_imbalance_level2}, "
            f"delta={self.delta}, "
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
                "imbalance_pressure_price": pa.float64(),
                "imbalance_support_price": pa.float64(),
                "pressure_levels": pa.uint64(),
                "support_levels":  pa.uint64(),
                "point_of_control": pa.float64(),
                "top_imbalance_level1": pa.float64(),
                "top_imbalance_level2": pa.float64(),
                "bottom_imbalance_level1": pa.float64(),
                "bottom_imbalance_level2": pa.float64(),
                "delta":pa.float64(),
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
            imbalance_pressure_price=values["imbalance_pressure_price"],
            imbalance_support_price=values["imbalance_support_price"],
            pressure_levels=values["pressure_levels"],
            support_levels=values["support_levels"],
            point_of_control=values["point_of_control"],
            top_imbalance_level1=values["top_imbalance_level1"],
            top_imbalance_level2=values["top_imbalance_level2"], 
            bottom_imbalance_level1=values["bottom_imbalance_level1"],
            bottom_imbalance_level2=values["bottom_imbalance_level2"],
            delta=values["delta"],
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
            "imbalance_pressure_price": obj.imbalance_pressure_price,
            "imbalance_support_price": obj.imbalance_support_price,
            "pressure_levels": obj.pressure_levels,
            "support_levels": obj.support_levels,
            "point_of_control": obj.point_of_control,
            "top_imbalance_level1": obj.top_imbalance_level1,
            "top_imbalance_level2": obj.top_imbalance_level2,
            "bottom_imbalance_level1": obj.bottom_imbalance_level1,
            "bottom_imbalance_level2": obj.bottom_imbalance_level2,
            "delta":obj.delta,
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
    def imbalance_pressure_price(self) -> double:
        """
        Return the imbalance_pressure_price.

        Returns
        -------
        double

        """
        return self.imbalance_pressure_price

    @property
    def imbalance_support_price(self) -> double:
        """
        Return the imbalance_support_price.

        Returns
        -------
        double

        """
        return self.imbalance_support_price

    @property
    def pressure_levels(self) -> int:
        """
        Return the pressure_levels.

        Returns
        -------
        double

        """
        return self.pressure_levels

    @property
    def support_levels(self) -> int:
        """
        Return the support_levels.

        Returns
        -------
        double

        """
        return self.support_levels

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
    def top_imbalance_level1(self) -> double:
        """
        Return the top_imbalance_level1.

        Returns
        -------
        double

        """
        return self.top_imbalance_level1

    @property
    def top_imbalance_level2(self) -> double:
        """
        Return the top_imbalance_level2.

        Returns
        -------
        double

        """
        return self.top_imbalance_level2

    @property
    def bottom_imbalance_level1(self) -> double:
        """
        Return the bottom_imbalance_level1.

        Returns
        -------
        double

        """
        return self.bottom_imbalance_level1

    @property
    def bottom_imbalance_level2(self) -> double:
        """
        Return the bottom_imbalance_level2.

        Returns
        -------
        double

        """
        return self.bottom_imbalance_level2

    @property
    def delta(self) -> double:
        """
        Return the delta.

        Returns
        -------
        double

        """
        return self.delta

