from rdkit import Chem


class PretrainGNNsAtomFeaturizer(object):
    """Atom featurizer used by pretrain-gnns."""

    def __init__(self):
        self._atomic_num_types = list(range(1, 119))
        self._chiral_types = [
            Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
            Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
            Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
            Chem.rdchem.ChiralType.CHI_OTHER
        ]

    def __call__(self, atom):
        return self.featurize_atom(atom)

    @property
    def feature_dim(self):
        atom = Chem.MolFromSmiles('CC').GetAtomWithIdx(0)
        return len(self(atom))

    def featurize_atom(self, atom):
        atom_feature = [
            self._atomic_num_types.index(atom.GetAtomicNum()),
            self._chiral_types.index(atom.GetChiralTag())
        ]
        return atom_feature


class PretrainGNNsBondFeaturizer(object):
    """Bond featurizer used by pretrain-gnns."""

    def __init__(self):
        self._bond_types = [
            Chem.rdchem.BondType.SINGLE,
            Chem.rdchem.BondType.DOUBLE,
            Chem.rdchem.BondType.TRIPLE,
            Chem.rdchem.BondType.AROMATIC
        ]
        self._bond_dir_types = [
            Chem.rdchem.BondDir.NONE,
            Chem.rdchem.BondDir.ENDUPRIGHT,
            Chem.rdchem.BondDir.ENDDOWNRIGHT
        ]

    def __call__(self, atom):
        return self.featurize_bond(atom)

    @property
    def feature_dim(self):
        bond = Chem.MolFromSmiles('CC').GetBondWithIdx(0)
        return len(self(bond))

    def featurize_bond(self, bond):
        bond_feature = [
            self._bond_types.index(bond.GetBondType()),
            self._bond_dir_types.index(bond.GetBondDir())
        ]
        return bond_feature
