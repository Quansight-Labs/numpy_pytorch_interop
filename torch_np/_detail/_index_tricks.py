"""
Vendored objects from numpy.lib.index_tricks
"""
__all__ = ["index_exp", "s_"]


class IndexExpression:
    """
    Written by Konrad Hinsen <hinsen@cnrs-orleans.fr>
    last revision: 1999-7-23

    Cosmetic changes by T. Oliphant 2001
    """

    def __init__(self, maketuple):
        self.maketuple = maketuple

    def __getitem__(self, item):
        if self.maketuple and not isinstance(item, tuple):
            return (item,)
        else:
            return item


index_exp = IndexExpression(maketuple=True)
s_ = IndexExpression(maketuple=False)
