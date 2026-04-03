__all__ = ['ScaleSC', 
           'clusters_merge',
           'find_markers', 
           'AnnDataBatchReader', 
           'check_nonnegative_integers',
           'correct_leiden',
           'write_to_disk',
           'gc',
           '__version__']
__version__ = '0.1.0'

import logging
from scalesc.pp import ScaleSC
from scalesc.util import AnnDataBatchReader, check_nonnegative_integers, correct_leiden, write_to_disk, gc
from scalesc.trim_merge_marker import clusters_merge, find_markers

logger = logging.getLogger("scaleSC")
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)
