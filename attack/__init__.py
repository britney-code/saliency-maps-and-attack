"""
@author: Britney(wanqiang512)
@software: PyCharm
@file: __init__.py
@time: 2023/10/15 23:15
"""
from attack import AAAM, FIA, fgsm, PGD, RPA, NAA, ssa, taig, tifgsm, SFVA, difgsm, sim, \
   mifgsm, pifgsm, admix

try:
    from .version import __version__
except ImportError:
    pass
