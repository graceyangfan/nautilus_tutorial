

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

from nautilus_trader.model.data.bar cimport Bar


cdef class OrderFlowBar(Bar):

    cdef int     pressure_levels 
    cdef int     support_levels 
    cdef double  bottom_imbalance 
    cdef double  bottom_imbalance_price 
    cdef double  middle_imbalance 
    cdef double  middle_imbalance_price 
    cdef double  top_imbalance 
    cdef double  top_imbalance_price 
    cdef double  point_of_control 
    cdef double  poc_imbalance    
    cdef double  delta 
    cdef double  value_delta 
    cdef bint    up_bar
    cdef int     tag


    @staticmethod
    cdef OrderFlowBar from_dict_c(dict values)

    @staticmethod
    cdef dict to_dict_c(OrderFlowBar obj)

