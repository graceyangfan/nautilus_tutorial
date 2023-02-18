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

from libc.stdint cimport uint8_t
from libc.stdint cimport uint64_t


from nautilus_trader.model.data.bar cimport BarType
from nautilus_trader.model.data.tick cimport TradeTick
from nautilus_trader.model.enums_c cimport AggressorSide
from nautilus_trader.model.data.imbalance_bar cimport ImbalanceBar
from nautilus_trader.model.objects cimport Price
from nautilus_trader.model.objects cimport Quantity
from nautilus_trader.indicators.zscore cimport Zscore


cdef class ImbalanceBarBuilder:
    cdef BarType _bar_type
    cdef Zscore zscore 
    cdef readonly uint8_t price_precision
    """The price precision for the builders instrument.\n\n:returns: `uint8`"""
    cdef readonly uint8_t size_precision
    """The size precision for the builders instrument.\n\n:returns: `uint8`"""
    cdef readonly bint initialized
    """If the builder is initialized.\n\n:returns: `bool`"""
    cdef readonly uint64_t ts_last
    """The UNIX timestamp (nanoseconds) when the builder last updated.\n\n:returns: `uint64_t`"""
    cdef readonly int count
    """The builders current update count.\n\n:returns: `int`"""
    cdef readonly object on_bar
    """The on_bar callback function.\n\n:returns: `object`"""
    cdef double threshold
    """The threshold for generate new bars.\n\n:returns: `double`"""
    cdef double value_level
    """The value level for trade volume classify.\n\n:returns: `double`"""
    cdef bint _partial_set
    cdef Price last_price
    cdef int tick_direction 
    cdef Price _last_close
    cdef Price _open
    cdef Price _high
    cdef Price _low
    cdef Price _close
    cdef Quantity volume
    cdef double small_buy_value
    cdef double big_buy_value
    cdef double small_sell_value
    cdef double big_sell_value

    cpdef void set_partial(self, ImbalanceBar partial_bar) except *
    cpdef void update(
        self, 
        Price price, 
        Quantity size, 
        AggressorSide aggressor_side,
        uint64_t ts_event
    ) except *
    cpdef void reset(self) except *
    cpdef ImbalanceBar build(self, uint64_t ts_event)
    cpdef void handle_trade_tick(self, TradeTick tick) except *
    cdef void _apply_update(
        self, 
        Price price, 
        Quantity size, 
        AggressorSide aggressor_side, 
        uint64_t ts_event
    ) except *