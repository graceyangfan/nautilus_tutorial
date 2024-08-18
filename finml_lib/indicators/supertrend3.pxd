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

from nautilus_trader.indicators.atr cimport AverageTrueRange
from nautilus_trader.indicators.average.ema cimport ExponentialMovingAverage
from nautilus_trader.indicators.base.indicator cimport Indicator
from nautilus_trader.model.data cimport Bar


cdef class SuperTrend3(Indicator):
    cdef AverageTrueRange _atr

    cdef readonly int period
    """The window period.\n\n:returns: `int`"""
    cdef readonly double k_multiplier
    """The k multiplier.\n\n:returns: `double`"""
    cdef readonly bint use_close
    """The zigzag use_close param.\n\n:returns: `double`"""
    cdef readonly double _previous_upper_band
    """The previous value of the upper channel.\n\n:returns: `double`"""
    cdef readonly double _previous_lower_band
    """The previous value of the lower channel.\n\n:returns: `double`"""
    cdef readonly double _current_upper_band
    """The current value of the upper channel.\n\n:returns: `double`"""
    cdef readonly double _current_lower_band
    """The current value of the lower channel.\n\n:returns: `double`"""
    cdef readonly double direction
    """The current direction.\n\n:returns: `double`"""
    cdef readonly double previous_direction
    """The previous direction.\n\n:returns: `double`"""
    cdef readonly bint change_direction
    """The value shows that whether the supertrend direction has changed.\n\n:returns: `bint`"""
    cpdef void handle_bar(self, Bar bar) except *
    cpdef void update_raw(self, double high, double low, double close) except *
