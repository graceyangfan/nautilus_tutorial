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

from enum import Enum
from enum import unique

@unique
class Mark(Enum):
    DI = 0 ##bottom
    DING = 1  ##top 

@unique
class Direction(Enum):
    UP = 0
    DOWN = 1
    OSCILLATION= 2 

@unique
class BIType(Enum):
    OLD = 0
    NEW = 1 
    TB = 2  ##top and bottom 

@unique 
class LineType(Enum):
    BI = 0 
    XD = 1  

@unique 
class ZSProcessType(Enum):
    INTERATE = 0  ##interate 
    INSIDE = 1 

@unique 
class SupportType(Enum):
    HL = 0  ## high and low 
    TB = 1  ##top and bottom  

@unique 
class DivergenceType(Enum):
    BI = 0 
    XD = 1 
    ZSD = 2 
    OSCILLATION = 3 
    TREND = 4  

@unique 
class TradePointType(Enum):
    OneBuy = 0
    OneSell = 1 
    TwoBuy = 2 
    TwoSell = 3 
    ThreeBuy = 4 
    ThreeSell = 5 
    L2Buy = 6 
    L2Sell = 7 
    L3Buy = 8 
    L3Sell = 9 
