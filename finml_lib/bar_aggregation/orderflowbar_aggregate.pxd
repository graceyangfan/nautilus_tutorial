# Class definition for ValueImbalanceBarAggregator, which inherits from ImbalanceBarBuilder
cdef class ValueImbalanceBarAggregator(ImbalanceBarBuilder):
    """
    ValueImbalanceBarAggregator extends ImbalanceBarBuilder and aggregates bars based on cumulative value.

    Parameters
    ----------
    instrument : Instrument
        The instrument for the aggregator.
    bar_type : BarType
        The bar type for the aggregator.
    on_bar : Callable
        The callback function for when a bar is completed.
    """
    # Constructor
    def __init__(
        self,
        Instrument instrument not None,
        BarType bar_type not None,
        on_bar not None: Callable,
    ):
        # Call the constructor of the base class (ImbalanceBarBuilder)
        super().__init__(
            instrument=instrument,
            bar_type=bar_type,
            on_bar=on_bar,
        )

        # Initialize cumulative value attribute
        self._cum_value = Decimal(0)

    # Method to
