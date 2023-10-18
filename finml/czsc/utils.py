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


from tracemalloc import start
from typing import List
from nautilus_trader.czsc.base_object import BI, LINE, MACD_INFOS
from nautilus_trader.czsc.enums import Direction 
from nautilus_trader.czsc.twist import Twist
from nautilus_trader.czsc.base_object import ZS 
import matplotlib.pyplot as plt 
import numpy as np 


def judge_macd_back_zero(
    twist: Twist, 
    zs: ZS
) -> int:
    """
    macd back to zero max times 
    """
    zs_macd_info = cal_zs_macd_infos(zs, twist)
    if zs.lines[0].direction_type == Direction.DOWN:
        return max(zs_macd_info.dem_down_cross_num, zs_macd_info.dif_down_cross_num)
    if zs.lines[0].direction_type == Direction.UP:
        return max(zs_macd_info.dem_up_cross_num, zs_macd_info.dif_up_cross_num)
    return 0

def last_confirm_line(lines: List[LINE]):
    """
    get last done bi 
    """
    for line in lines[::-1]:
        if line.is_confirm:
            return line 
    return None


def get_zs_with_line(twist: Twist,line:LINE):
    zs = None 
    for _zs in twist.zss:
        if _zs.lines[-1].index == line.index:
            zs = _zs 
    if zs:
        return zs 

def cal_zs_macd_infos(zs: ZS, twist: Twist) -> MACD_INFOS:
    """
    get ZS info 
    """
    infos = MACD_INFOS()
    start_index = zs.start.middle_twist_bar.b_index
    end_index = zs.end.middle_twist_bar.b_index  + 1
    dem = np.array(twist.macd_dem[start_index:end_index])
    dif = np.array(twist.macd_dif[start_index:end_index])
    if len(dem) < 2 or len(dif) < 2:
        return infos
    zero = np.zeros(len(dem))

    infos.dif_up_cross_num = len(up_cross(dif, zero))
    infos.dif_down_cross_num = len(down_cross(dif, zero))
    infos.dem_up_cross_num = len(up_cross(dem, zero))
    infos.dem_down_cross_num = len(down_cross(dem, zero))
    infos.gold_cross_num = len(up_cross(dif, dem))
    infos.die_cross_num = len(down_cross(dif, dem))
    infos.last_dif = dif[-1]
    infos.last_dem = dem[-1]
    return infos


def up_cross(one_list: np.array, two_list: np.array):
    """
    macd up cross 
    """
    assert len(one_list) == len(two_list)
    if len(one_list) < 2:
        return []
    cross = []
    for i in range(1, len(two_list)):
        if one_list[i - 1] < two_list[i - 1] and one_list[i] > two_list[i]:
            cross.append(i)
    return cross


def down_cross(one_list: np.array, two_list: np.array):
    """
    macd down cross 
    """
    assert len(one_list) == len(two_list)
    if len(one_list) < 2:
        return []
    cross = []
    for i in range(1, len(two_list)):
        if one_list[i - 1] > two_list[i - 1] and one_list[i] < two_list[i]:
            cross.append(i)
    return cross


def bi_pause(bi: BI, twist: Twist):
    """
    BI is pause 
    """
    if not bi.is_confirm:
        return False
    last_price = twist.newbars[-1].close 
    if bi.direction_type == Direction.UP and last_price < bi.end.twist_bars[-1].low: 
        return True
    elif bi.direction_type == Direction.DOWN and last_price > bi.end.twist_bars[-1].high:
        return True

    return False


def plot_bis(bis: List):
    for bi in bis:
        if bi.is_confirm:
            plt.plot([bi.ts_opened,bi.ts_closed],
                     [bi.start.middle_twist_bar.close, bi.end.middle_twist_bar.close], color="red") 
        else:
            plt.plot([bi.ts_opened,bi.ts_closed],
                     [bi.start.middle_twist_bar.close, bi.end.middle_twist_bar.close], color="blue")
            
def plot_zss(zss: List):
    for zs in zss:
        if zs.is_confirm:
            plt.plot([zs.ts_opened,zs.ts_opened],[zs.dd,zs.gg], color="red")
            plt.plot([zs.ts_opened,zs.ts_closed],[zs.dd,zs.dd], color="red")
            plt.plot([zs.ts_opened,zs.ts_closed],[zs.gg,zs.gg], color="red")
            plt.plot([zs.ts_closed,zs.ts_closed],[zs.dd,zs.gg], color="red")
        else:
            plt.plot([zs.ts_opened,zs.ts_opened],[zs.dd,zs.gg], color="blue")
            plt.plot([zs.ts_opened,zs.ts_closed],[zs.dd,zs.dd], color="blue")
            plt.plot([zs.ts_opened,zs.ts_closed],[zs.gg,zs.gg], color="blue")
            plt.plot([zs.ts_closed,zs.ts_closed],[zs.dd,zs.gg], color="blue")

def plot_xd(xds: List):
    for xd in xds:
        if xd.is_confirm:
            plt.plot([xd.start_line.ts_opened,xd.end_line.ts_closed],[xd.start_line.start.middle_twist_bar.close,xd.end_line.end.middle_twist_bar.close], color="red")
        else:
            plt.plot([xd.start_line.ts_opened,xd.end_line.ts_closed],[xd.start_line.start.middle_twist_bar.close,xd.end_line.end.middle_twist_bar.close], color="blue")
