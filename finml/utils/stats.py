from scipy.stats import rv_continuous
class KDERv(rv_continuous):

    def __init__(self, kde, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._kde = kde

    def _pdf(self, x, *args):
        return self._kde.pdf(x)