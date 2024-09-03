from __future__ import annotations

import os

import numpy as np
import rootutils
from beartype import beartype
from beartype.door import is_bearable
from Bio.PDB.Atom import Atom, DisorderedAtom
from Bio.PDB.Chain import Chain
from Bio.PDB.Residue import DisorderedResidue, Residue
from jaxtyping import Bool, Float, Int, Shaped, jaxtyped
from torch import Tensor

from alphafold3_pytorch.utils.pylogger import RankedLogger
from alphafold3_pytorch.utils.utils import always, identity

logger = RankedLogger(__name__, rank_zero_only=True)

# environment

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


# NOTE: `jaxtyping` is a misnomer, works for PyTorch as well


class TorchTyping:
    """Torch typing."""

    def __init__(self, abstract_dtype):
        self.abstract_dtype = abstract_dtype

    def __getitem__(self, shapes: str):
        """Get item."""
        return self.abstract_dtype[Tensor, shapes]


Shaped = TorchTyping(Shaped)
Float = TorchTyping(Float)
Int = TorchTyping(Int)
Bool = TorchTyping(Bool)

# helper type aliases

IntType = int | np.int32 | np.int64
AtomType = Atom | DisorderedAtom
ResidueType = Residue | DisorderedResidue
ChainType = Chain
TokenType = AtomType | ResidueType

# NOTE: use env variable `TYPECHECK` (which is set by `rootutils` above using `.env`) to control whether to use `beartype` + `jaxtyping`
# NOTE: use env variable `DEBUG` to control whether to print debugging information

should_typecheck = os.environ.get("TYPECHECK", False)
IS_DEBUGGING = os.environ.get("DEBUG", False)

typecheck = jaxtyped(typechecker=beartype) if should_typecheck else identity

beartype_isinstance = is_bearable if should_typecheck else always(True)

if should_typecheck:
    logger.info("Type checking is enabled.")
if IS_DEBUGGING:
    logger.info("Debugging is enabled.")

__all__ = [
    beartype_isinstance,
    Bool,
    Float,
    Int,
    Shaped,
    should_typecheck,
    typecheck,
    IS_DEBUGGING,
]
