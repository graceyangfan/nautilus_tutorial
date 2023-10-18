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

from typing import List 
from nautilus_trader.czsc.base_object import LINE, LOW_LEVEL_TREND, ZS
from nautilus_trader.czsc.enums import Direction, LineType
from nautilus_trader.czsc.twist import Twist 

class MultiLevelAnalyse:
    """
    Analyse high level and low level twist together 
    """
    def __init__(
        self,
        high_level: Twist,
        low_level: Twist,
    ):
        self.high_level: Twist = high_level 
        self.low_level: Twist = low_level 

    def low_level_trend(
        self, 
        up_line: LINE, 
        low_line_type=LineType.BI
    ) -> LOW_LEVEL_TREND:
        """
        query low level info  with up lines 
        """
        low_lines = self._query_low_lines(up_line, low_line_type)
        low_zss = self._query_low_zss(low_lines, low_line_type)
        trend_divergence_info = self._query_trend_and_divergence(low_lines, low_zss, low_line_type)

        low_level_trend = LOW_LEVEL_TREND(low_zss, low_lines)
        low_level_trend.zs_num = len(low_zss)
        low_level_trend.line_num = len(low_lines)
        low_level_trend.last_line = low_lines[-1] if len(low_lines) > 0 else None
        low_level_trend.trend = trend_divergence_info['trend']
        low_level_trend.oscillation = trend_divergence_info['oscillation']
        low_level_trend.line_divergence = trend_divergence_info['line_divergence']
        low_level_trend.trend_divergence = trend_divergence_info['trend_divergence']
        low_level_trend.oscillation_divergence = trend_divergence_info['oscillation_divergence']
        low_level_trend.divergence_line = trend_divergence_info['divergence_line']

        return low_level_trend 

    def up_bi_low_level_trend(self) -> LOW_LEVEL_TREND:
        """
        use high level bi to query low level info 
        """
        last_bi = self.high_level.bis[-1]
        return self.low_level_trend(last_bi, LineType.BI)

    def up_xd_low_level_trend(self) -> LOW_LEVEL_TREND:
        """
        query low level info with high level xd 
        """
        last_xd = self.high_level.xds[-1] 
        return self.low_level_trend(last_xd, LineType.BI)

    def _query_low_lines(
        self,
        up_line: LINE, 
        query_line_type=LineType.BI
    ):
        """
        query low level lines include in high lines 
        """
        start_date = up_line.start.middle_twist_bar.ts_opened 
        end_date = up_line.end.middle_twist_bar.ts_closed 

        low_lines: List[LINE] = []
        find_lines = self.low_level.bis if query_line_type == LineType.BI else self.low_level.xds 
        for _line in find_lines:
            if _line.end.middle_twist_bar.ts_opened < start_date:
                continue
            if end_date is not None and _line.start.middle_twist_bar.ts_closed  > end_date:
                break
            if len(low_lines) == 0 and _line.direction_type != up_line.direction_type:
                continue
            low_lines.append(_line)
        if len(low_lines) > 0 and low_lines[-1].direction_type != up_line.direction_type:
            low_lines.pop() 

        return low_lines 

    def _query_low_zss(
        self, 
        low_lines: List[LINE], 
        zs_type=LineType.BI
    ):
        """
        create zs with low level lines 
        """
        low_zss = self.low_level.create_inside_zs(zs_type, low_lines)
        return low_zss

    def _query_trend_and_divergence(
        self,
        low_lines: List[LINE], 
        low_zss: List[ZS], 
        low_line_type=LineType.BI
    ):
        """
        compute if is ZS_divergence based with low level lines and zs 
        """
        trend = False
        oscillation = False
        trend_divergence = False
        oscillation_divergence = False
        line_divergence = False

        # if is divergence 
        if len(low_lines) >= 3:
            one_line = low_lines[-3]
            two_line = low_lines[-1]
            if two_line.direction_type == Direction.UP and one_line.direction_type == Direction.UP \
                    and two_line.high > one_line.high and two_line.low > one_line.low \
                    and self.low_level.compare_power_divergence(one_line.power, two_line.power):
                line_divergence = True
            elif two_line.direction_type == Direction.DOWN  and one_line.direction_type == Direction.DOWN \
                    and two_line.low < one_line.low and two_line.high < one_line.high \
                    and self.low_level.compare_power_divergence(one_line.power, two_line.power):
                line_divergence = True

        if len(low_zss) == 0:
            return {'trend': trend, 'oscillation': oscillation, 'line_divergence': line_divergence, 'trend_divergence': trend_divergence, 'oscillation_divergence': oscillation_divergence, 'divergence_line': None}

        # osciliation divergecne 
        oscillation = True if low_zss[-1].direction_type in [Direction.UP, Direction.DOWN] else False
        oscillation_divergence  = self.low_level.divergence_oscillation(low_zss[-1], low_zss[-1].lines[-1])

        # trend divergence 
        if len(low_zss) >= 2:
            trend = self.low_level.zss_is_trend(low_zss[-2], low_zss[-1])
            if not trend:
                trend_divergence = False 
            else:
                trend_divergence = self.low_level.compare_power_divergence(low_zss[-2].lines[-1].power,low_zss[-1].lines[-1].power)

        divergence_line = None
        if oscillation_divergence or trend_divergence:
            divergence_line = low_zss[-1].lines[-1]

        return {'trend': trend, 'oscillation': oscillation, 'line_divergence': line_divergence, 'trend_divergence': trend_divergence, 'oscillation_divergence': oscillation_divergence, 'divergence_line': divergence_line}