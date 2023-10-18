import re
from ops import Operators,register_all_ops
from base import Feature

class ExprEngine:
    @classmethod
    def init(cls):
        register_all_ops()

    @staticmethod
    def parse_field(field):
        # Following patterns will be matched:
        # - $close -> Feature("close")
        # - $close5 -> Feature("close5")
        # - $open+$close -> Feature("open")+Feature("close")
        if not isinstance(field, str):
            field = str(field)
        return re.sub(r"\$(\w+)", r'Feature("\1")', re.sub(r"(\w+\s*)\(", r"Operators.\1(", field))

    @staticmethod
    def get_expression(feature):
        feature = ExprEngine.parse_field(feature)
        print(feature)
        try:
            expr = eval(feature)
        except:
            print('error',feature)
            raise
        return expr

