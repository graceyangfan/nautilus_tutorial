

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

from nautilus_trader.model.data cimport Bar


cdef class ImbalanceBar(Bar):
    cdef double big_buy_ratio
    cdef double big_net_buy_ratio
    cdef double big_buy_power
    cdef double big_net_buy_power
    cdef double value_delta 
    cdef int    tag

    @staticmethod
    cdef ImbalanceBar from_dict_c(dict values)

    @staticmethod
    cdef dict to_dict_c(Bar obj)

