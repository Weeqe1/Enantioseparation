# ==============================================================================
# Feature Calculation Module
# ==============================================================================
# This module provides functionality for molecular feature calculation, including:
# - Molecular descriptor computation using Mordred
# - Graph representation of molecules using PyTorch Geometric
# - 3D molecular structure generation and optimization
# - Dataset construction for machine learning models
# ==============================================================================

# Standard Library Imports
import os

# Third-party Scientific Computing Libraries
import numpy as np
import pandas as pd

# Progress Bar Library
from tqdm import tqdm

# PyTorch Libraries
import torch
from torch_geometric.data import Data

# Chemistry Libraries - RDKit
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdchem

# Molecular Descriptor Library
from mordred import Calculator, descriptors

# Local Module Imports
from .id_names import atom_id_names, bond_id_names

# ==============================================================================
# Global Constants
# ==============================================================================

# Master calculator for all molecular descriptors using Mordred
# ignore_3D=False enables 3D descriptor calculations
MASTER_CALCULATOR = Calculator(descriptors, ignore_3D=False)


# ==============================================================================
# Utility Functions
# ==============================================================================

def rdchem_enum_to_list(values):
    """
    Convert an RDKit rdchem enum to a Python list.

    This function extracts all possible values from an RDKit enumeration
    and returns them as a list for use in feature vocabularies.

    Args:
        values (rdchem.Enum): An RDKit enumeration object containing 
                             predefined chemical property values

    Returns:
        list: A list containing all values from the enum in order

    Example:
        >>> bond_types = rdchem_enum_to_list(rdchem.BondType.values)
        >>> # Returns list of all bond types like [SINGLE, DOUBLE, TRIPLE, ...]
    """
    return [values[i] for i in range(len(values))]


def safe_index(alist, elem):
    """
    Safely retrieve the index of an element in a list with fallback.

    This function attempts to find the index of an element in a list.
    If the element is not found, it returns the last index as a fallback
    (typically used for 'misc' or 'other' category in feature vocabularies).

    Args:
        alist (list): The list to search in
        elem: The element to find the index for

    Returns:
        int: The index of the element if found, otherwise the last index 
             of the list (len(alist) - 1)

    Example:
        >>> vocab = ['A', 'B', 'C', 'misc']
        >>> safe_index(vocab, 'B')  # Returns 1
        >>> safe_index(vocab, 'X')  # Returns 3 (index of 'misc')
    """
    try:
        return alist.index(elem)
    except ValueError:
        return len(alist) - 1


def get_atom_feature_dims(list_acquired_feature_names):
    """
    Calculate the dimensions of atom features based on vocabulary sizes.

    This function determines how many possible values each atom feature can take
    by looking up the vocabulary size for each feature name.

    Args:
        list_acquired_feature_names (list): List of atom feature names 
                                          (e.g., ['atomic_num', 'degree'])

    Returns:
        list: List of integers representing the vocabulary size for each feature

    Example:
        >>> get_atom_feature_dims(['atomic_num', 'degree'])
        >>> # Returns [119, 12] (atomic numbers 1-118 + misc, degrees 0-10 + misc)
    """
    return list(map(len, [CompoundKit.atom_vocab_dict[name] for name in list_acquired_feature_names]))


def get_bond_feature_dims(list_acquired_feature_names):
    """
    Calculate the dimensions of bond features with padding for self-loops.

    Similar to atom features, but adds 1 to each dimension to account for
    self-loop edges that don't have standard bond properties.

    Args:
        list_acquired_feature_names (list): List of bond feature names
                                          (e.g., ['bond_type', 'is_in_ring'])

    Returns:
        list: List of integers representing vocabulary size + 1 for each feature

    Example:
        >>> get_bond_feature_dims(['bond_type'])
        >>> # Returns [6] if bond_type vocab has 5 values (5 + 1 for self-loops)
    """
    list_bond_feat_dim = list(map(len, [CompoundKit.bond_vocab_dict[name] for name in list_acquired_feature_names]))
    return [_l + 1 for _l in list_bond_feat_dim]


def get_column_molecules_smiles(TRANSFER_TARGET):
    """
    Retrieve SMILES representation of chromatographic column molecules.

    This function maps chromatographic column names to their corresponding
    SMILES representations of the stationary phase molecules.

    Args:
        TRANSFER_TARGET (str): The name of the chromatographic column
                              (e.g., 'Cyclosil_B', 'CP_Chirasil_D_Val')

    Returns:
        str: SMILES representation of the column molecule, or 
             "Unknown TRANSFER_TARGET" if the column is not found

    Example:
        >>> get_column_molecules_smiles('Cyclosil_B')
        >>> # Returns the SMILES string for Cyclosil B stationary phase
    """
    # Dictionary mapping column names to their stationary phase SMILES
    column_molecules_dict = {
        'Cyclosil_B': 'CC(C)(C)[Si](C)(C)OC[C@@H]1C([O])[C@@H](OC)[C@@H](OC)[C@@H]([O])O1',
        'Cyclodex_B': 'OC[C@@H]1C([C@H]([C@H]([C@H](O1)[O])O)O)[O]',
        'HP_chiral_20β': 'OC[C@@H]1C([C@H]([C@H]([C@H](O1)[O])O)O)[O]',
        'CP_Cyclodextrin_β_2,3,6_M_19': 'OC[C@@H]1C([C@H]([C@H]([C@H](O1)[O])O)O)[O]',
        'CP_Chirasil_D_Val': 'CC(C)[C@H](NC(=O)C(C)(C)C)C(=O)O',
        'CP_Chirasil_Dex_CB': 'OC[C@@H]1C([C@H]([C@H]([C@H](O1)[O])O)O)[O]',
        'CP_Chirasil_L_Val': 'CC(C)[C@@H](NC(=O)C(C)(C)C)C(=O)O'
    }

    return column_molecules_dict.get(TRANSFER_TARGET, "Unknown TRANSFER_TARGET")


def calculate_descriptors(mol, column_name, molecule_type):
    """
    Calculate molecular descriptors using Mordred based on SHAP feature selection.

    This function computes molecular descriptors for either chiral molecules or 
    column molecules using a pre-selected set of features identified through 
    SHAP analysis. It handles error cases gracefully by returning zero values.

    Args:
        mol (rdkit.Chem.Mol): RDKit molecule object to calculate descriptors for
        column_name (str): Name of the chromatographic column (e.g., 'Cyclosil_B')
        molecule_type (str): Type of molecule - either 'chiral' or 'column'

    Returns:
        list: List of calculated descriptor values in predefined order. 
              Returns list of zeros if molecule is invalid or no matching 
              descriptors are found.

    Example:
        >>> mol = Chem.MolFromSmiles('CCO')
        >>> descriptors = calculate_descriptors(mol, 'Cyclosil_B', 'chiral')
        >>> # Returns list of descriptor values for the molecule
    """
    # Path to the feature selection file generated by SHAP analysis
    features_file = 'Output/SHAP_Results/comparison_of_selected_features.csv'

    try:
        # Load pre-selected features from SHAP analysis results
        selected_features_df = pd.read_csv(features_file)
    except FileNotFoundError:
        print(f"Error: feature file'{features_file}'not found.")
        # Create empty DataFrame with expected columns to prevent downstream errors
        selected_features_df = pd.DataFrame(columns=['feature', 'Column'])

    # Determine feature prefix based on molecule type
    if molecule_type == 'chiral':
        prefix = 'mol_'  # Prefix for chiral molecule features
    elif molecule_type == 'column':
        prefix = 'columo_'  # Prefix for column molecule features
    else:
        raise ValueError("Molecule_type must be 'chiral' or 'column'")

    # Filter features for specific column and molecule type
    column_features = selected_features_df[selected_features_df['Column'] == column_name]
    specific_features = column_features[column_features['feature'].str.startswith(prefix, na=False)]

    # Get sorted list of feature names to ensure consistent ordering
    descriptor_names_with_prefix = sorted(specific_features['feature'].tolist())

    # Remove prefix to get actual Mordred descriptor names
    descriptor_names = [name.replace(prefix, '', 1) for name in descriptor_names_with_prefix]

    # Initialize data dictionary with zeros to handle calculation failures gracefully
    data = {name: 0.0 for name in descriptor_names_with_prefix}

    # Return zeros if molecule is None or no descriptors to calculate
    if mol is None or not descriptor_names:
        if not descriptor_names:
            print(f"Warning: the'{molecule_type}'type descriptor for column'{column_name}' was not found. Returns a zero value.")
        return list(data.values())

    try:
        # Find required descriptor objects from the master calculator
        selected_descriptor_objects = [
            desc for desc in MASTER_CALCULATOR.descriptors
            if str(desc) in descriptor_names
        ]

        # Warn and return zeros if no matching descriptor objects found
        if not selected_descriptor_objects:
            print(f"Warning: cannot find any of the specified descriptors in Mordred: {descriptor_names}. Returns a zero value.")
            return list(data.values())

        # Create specialized calculator with selected descriptors only
        subset_calc = Calculator(selected_descriptor_objects, ignore_3D=False)

        # Mordred typically requires 3D coordinates - ensure they exist
        mol_3d = Chem.AddHs(mol)  # Add hydrogens for better 3D generation
        if mol_3d.GetNumConformers() == 0:
            # Generate 3D coordinates if none exist
            AllChem.EmbedMolecule(mol_3d, AllChem.ETKDG())

        # Calculate descriptors using the specialized calculator
        mord_results = subset_calc(mol_3d)

        # Fill data dictionary with calculated values, handling errors gracefully
        for i, desc_obj in enumerate(subset_calc.descriptors):
            desc_name = str(desc_obj)
            full_feature_name = prefix + desc_name
            if full_feature_name in data:  # Check if this feature is desired
                value = mord_results[i]
                # Convert errors/NaN to float, then fill with 0
                try:
                    data[full_feature_name] = float(value)
                    if np.isnan(data[full_feature_name]):
                        data[full_feature_name] = 0.0
                except (ValueError, TypeError):
                    data[full_feature_name] = 0.0

    except Exception as e:
        print(f"Error calculating Mordred descriptor for column'{column_name}'and type'{molecule_type}': {e}")
        # Return zeros on any calculation error
        return list(data.values())

    # Return descriptor values in predefined order
    return [data[name] for name in descriptor_names_with_prefix]


# ==============================================================================
# Main Feature Classes
# ==============================================================================

class CompoundKit(object):
    """
    Comprehensive toolkit for managing molecular compounds and their features.

    This class provides vocabulary dictionaries and methods for extracting
    both categorical and continuous features from atoms and bonds in molecules.
    It serves as the foundation for molecular graph construction.
    """

    # ==============================================================================
    # Feature Vocabularies
    # ==============================================================================

    # Vocabulary dictionary for atom features with all possible values
    atom_vocab_dict = {
        "atomic_num": list(range(1, 119)) + ['misc'],  # Atomic numbers 1-118 + miscellaneous
        "chiral_tag": rdchem_enum_to_list(rdchem.ChiralType.values) + ['misc'],  # Chirality types
        "degree": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 'misc'],  # Number of bonds to atom
        "explicit_valence": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 'misc'],  # Explicit valence
        "formal_charge": [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 'misc'],  # Formal charge
        "hybridization": rdchem_enum_to_list(rdchem.HybridizationType.values) + ['misc'],  # Hybridization states
        "implicit_valence": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 'misc'],  # Implicit valence
        "is_aromatic": [0, 1, 'misc'],  # Aromaticity indicator (binary)
        "total_numHs": [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],  # Total number of hydrogens
        'num_radical_e': [0, 1, 2, 3, 4, 'misc'],  # Number of radical electrons
        'atom_is_in_ring': [0, 1, 2, 'misc'],  # Ring membership indicator
        'valence_out_shell': [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],  # Valence shell electrons
    }

    # Vocabulary dictionary for bond features with all possible values
    bond_vocab_dict = {
        "bond_dir": rdchem_enum_to_list(rdchem.BondDir.values) + ['misc'],  # Bond direction
        "bond_type": rdchem_enum_to_list(rdchem.BondType.values) + ['misc'],  # Bond types (single, double, etc.)
        "is_in_ring": [0, 1, 'misc'],  # Ring membership indicator
        'bond_stereo': rdchem_enum_to_list(rdchem.BondStereo.values) + ['misc'],  # Bond stereochemistry
        'is_conjugated': [0, 1, 'misc'],  # Conjugation indicator
    }

    # Names of continuous (floating-point) features for atoms and bonds
    atom_float_names = ["van_der_waals_radis", 'mass']  # Physical properties
    bond_float_names = ["bond_length"]  # Geometric property

    # Periodic table instance for accessing atomic properties
    period_table = Chem.GetPeriodicTable()

    # ==============================================================================
    # Atom Feature Extraction Methods
    # ==============================================================================

    @staticmethod
    def get_atom_value(atom, name):
        """
        Extract a specific atomic feature value by name.

        This method serves as a unified interface for accessing various atomic
        properties from RDKit atom objects.

        Args:
            atom (rdkit.Chem.Atom): RDKit atom object
            name (str): Name of the feature to extract

        Returns:
            The value of the requested atomic feature

        Raises:
            ValueError: If the feature name is not recognized

        Example:
            >>> atom = mol.GetAtomWithIdx(0)
            >>> atomic_num = CompoundKit.get_atom_value(atom, 'atomic_num')
        """
        if name == 'atomic_num':  # Atomic number (Z)
            return atom.GetAtomicNum()
        elif name == 'chiral_tag':  # Chirality classification
            return atom.GetChiralTag()
        elif name == 'degree':  # Number of bonded atoms
            return atom.GetDegree()
        elif name == 'explicit_valence':  # Explicitly specified valence
            return atom.GetExplicitValence()
        elif name == 'formal_charge':  # Formal charge on atom
            return atom.GetFormalCharge()
        elif name == 'hybridization':  # Hybridization state (sp, sp2, sp3, etc.)
            return atom.GetHybridization()
        elif name == 'implicit_valence':  # Implicitly calculated valence
            return atom.GetImplicitValence()
        elif name == 'is_aromatic':  # Aromaticity flag
            return int(atom.GetIsAromatic())
        elif name == 'mass':  # Atomic mass
            return int(atom.GetMass())
        elif name == 'total_numHs':  # Total hydrogen count
            return atom.GetTotalNumHs()
        elif name == 'num_radical_e':  # Unpaired electrons
            return atom.GetNumRadicalElectrons()
        elif name == 'atom_is_in_ring':  # Ring membership
            return int(atom.IsInRing())
        elif name == 'valence_out_shell':  # Outer shell electrons
            return CompoundKit.period_table.GetNOuterElecs(atom.GetAtomicNum())
        elif name == 'van_der_waals_radis':  # Van der Waals radius
            return CompoundKit.period_table.GetRvdw(atom.GetAtomicNum())
        else:
            raise ValueError(name)  # Unknown feature name

    @staticmethod
    def get_atom_feature_id(atom, name):
        """
        Get the vocabulary index for an atom's feature value.

        This method converts raw atomic feature values to indices that can
        be used for embedding or one-hot encoding in machine learning models.

        Args:
            atom (rdkit.Chem.Atom): RDKit atom object
            name (str): Name of the feature to get ID for

        Returns:
            int: Index of the feature value in the vocabulary

        Example:
            >>> atom_id = CompoundKit.get_atom_feature_id(atom, 'atomic_num')
            >>> # Returns index of atomic number in atomic_num vocabulary
        """
        assert name in CompoundKit.atom_vocab_dict, "%s not found in atom_vocab_dict" % name
        return safe_index(CompoundKit.atom_vocab_dict[name], CompoundKit.get_atom_value(atom, name))

    @staticmethod
    def get_atom_feature_size(name):
        """
        Get the vocabulary size for a specific atom feature.

        Args:
            name (str): Name of the atom feature

        Returns:
            int: Size of the vocabulary for this feature
        """
        assert name in CompoundKit.atom_vocab_dict, "%s not found in atom_vocab_dict" % name
        return len(CompoundKit.atom_vocab_dict[name])

    # ==============================================================================
    # Bond Feature Extraction Methods
    # ==============================================================================

    @staticmethod
    def get_bond_value(bond, name):
        """
        Extract a specific bond feature value by name.

        This method provides unified access to various bond properties
        from RDKit bond objects.

        Args:
            bond (rdkit.Chem.Bond): RDKit bond object
            name (str): Name of the feature to extract

        Returns:
            The value of the requested bond feature

        Raises:
            ValueError: If the feature name is not recognized
        """
        if name == 'bond_dir':  # Bond direction (up, down, none, etc.)
            return bond.GetBondDir()
        elif name == 'bond_type':  # Bond order (single, double, triple, aromatic)
            return bond.GetBondType()
        elif name == 'is_in_ring':  # Ring membership indicator
            return int(bond.IsInRing())
        elif name == 'is_conjugated':  # Part of conjugated system
            return int(bond.GetIsConjugated())
        elif name == 'bond_stereo':  # Stereochemistry (E, Z, none, etc.)
            return bond.GetStereo()
        else:
            raise ValueError(name)  # Unknown feature name

    @staticmethod
    def get_bond_feature_id(bond, name):
        """
        Get the vocabulary index for a bond's feature value.

        Similar to atom features, this converts bond properties to vocabulary indices.

        Args:
            bond (rdkit.Chem.Bond): RDKit bond object
            name (str): Name of the feature to get ID for

        Returns:
            int: Index of the feature value in the vocabulary
        """
        assert name in CompoundKit.bond_vocab_dict, "%s not found in bond_vocab_dict" % name
        return safe_index(CompoundKit.bond_vocab_dict[name], CompoundKit.get_bond_value(bond, name))

    @staticmethod
    def get_bond_feature_size(name):
        """
        Get the vocabulary size for a specific bond feature.

        Args:
            name (str): Name of the bond feature

        Returns:
            int: Size of the vocabulary for this feature
        """
        assert name in CompoundKit.bond_vocab_dict, "%s not found in bond_vocab_dict" % name
        return len(CompoundKit.bond_vocab_dict[name])

    # ==============================================================================
    # Feature Vector Construction Methods
    # ==============================================================================

    @staticmethod
    def atom_to_feat_vector(atom):
        """
        Convert an atom to a comprehensive feature dictionary.

        This method extracts all relevant atomic features and organizes them
        into a dictionary format suitable for further processing.

        Args:
            atom (rdkit.Chem.Atom): RDKit atom object

        Returns:
            dict: Dictionary containing all atomic features with their values

        Example:
            >>> atom_features = CompoundKit.atom_to_feat_vector(atom)
            >>> # Returns dict with keys like 'atomic_num', 'degree', etc.
        """
        atom_names = {
            # Categorical features (converted to vocabulary indices)
            "atomic_num": safe_index(CompoundKit.atom_vocab_dict["atomic_num"], atom.GetAtomicNum()),
            "chiral_tag": safe_index(CompoundKit.atom_vocab_dict["chiral_tag"], atom.GetChiralTag()),
            "degree": safe_index(CompoundKit.atom_vocab_dict["degree"], atom.GetTotalDegree()),
            "explicit_valence": safe_index(CompoundKit.atom_vocab_dict["explicit_valence"], atom.GetExplicitValence()),
            "formal_charge": safe_index(CompoundKit.atom_vocab_dict["formal_charge"], atom.GetFormalCharge()),
            "hybridization": safe_index(CompoundKit.atom_vocab_dict["hybridization"], atom.GetHybridization()),
            "implicit_valence": safe_index(CompoundKit.atom_vocab_dict["implicit_valence"], atom.GetImplicitValence()),
            "is_aromatic": safe_index(CompoundKit.atom_vocab_dict["is_aromatic"], int(atom.GetIsAromatic())),
            "total_numHs": safe_index(CompoundKit.atom_vocab_dict["total_numHs"], atom.GetTotalNumHs()),
            'num_radical_e': safe_index(CompoundKit.atom_vocab_dict['num_radical_e'], atom.GetNumRadicalElectrons()),
            'atom_is_in_ring': safe_index(CompoundKit.atom_vocab_dict['atom_is_in_ring'], int(atom.IsInRing())),
            'valence_out_shell': safe_index(CompoundKit.atom_vocab_dict['valence_out_shell'],
                                            CompoundKit.period_table.GetNOuterElecs(atom.GetAtomicNum())),
            # Continuous features (raw values)
            'van_der_waals_radis': CompoundKit.period_table.GetRvdw(atom.GetAtomicNum()),
            'mass': atom.GetMass(),
        }
        return atom_names  # Return the complete feature dictionary

    @staticmethod
    def get_atom_names(mol):
        """
        Extract feature vectors for all atoms in a molecule.

        This method processes all atoms in a molecule and returns their
        feature representations. It also computes Gasteiger partial charges.

        Args:
            mol (rdkit.Chem.Mol): RDKit molecule object

        Returns:
            list: List of feature dictionaries, one for each atom in the molecule

        Example:
            >>> all_atom_features = CompoundKit.get_atom_names(mol)
            >>> # Returns list where each element is a dict of atom features
        """
        atom_features_dicts = []
        # Compute Gasteiger partial charges for the entire molecule
        Chem.rdPartialCharges.ComputeGasteigerCharges(mol)

        # Extract features for each atom
        for i, atom in enumerate(mol.GetAtoms()):
            atom_features_dicts.append(CompoundKit.atom_to_feat_vector(atom))
        return atom_features_dicts


# ==============================================================================
# 3D Molecular Structure Class
# ==============================================================================

class Compound3DKit(object):
    """
    Toolkit for handling 3D molecular structures and geometric calculations.

    This class provides methods for generating 3D conformations, optimizing
    molecular geometries, and calculating geometric features like bond lengths
    and bond angles.
    """

    @staticmethod
    def get_atom_poses(mol, conf):
        """
        Extract 3D coordinates of all atoms from a molecular conformation.

        Args:
            mol (rdkit.Chem.Mol): RDKit molecule object
            conf (rdkit.Chem.Conformer): Molecular conformation containing 3D coordinates

        Returns:
            list: List of [x, y, z] coordinate lists for each atom

        Example:
            >>> positions = Compound3DKit.get_atom_poses(mol, conformer)
            >>> # Returns [[x1,y1,z1], [x2,y2,z2], ...] for each atom
        """
        atom_poses = []
        for i, atom in enumerate(mol.GetAtoms()):
            # Extract 3D position from conformer
            pos = conf.GetAtomPosition(i)
            atom_poses.append([pos.x, pos.y, pos.z])  # Convert to list format
        return atom_poses

    @staticmethod
    def get_MMFF_atom_poses(mol, numConfs=None):
        """
        Generate optimized 3D atom positions using MMFF force field.

        This method generates multiple conformations and selects the one with
        the lowest energy after MMFF optimization.

        Args:
            mol (rdkit.Chem.Mol): Input RDKit molecule object
            numConfs (int, optional): Number of conformations to generate

        Returns:
            tuple: (optimized_molecule, atom_positions)
                - optimized_molecule: RDKit molecule with optimized geometry
                - atom_positions: List of 3D coordinates for each atom

        Example:
            >>> opt_mol, positions = Compound3DKit.get_MMFF_atom_poses(mol, numConfs=10)
        """
        try:
            # Add hydrogens for better conformation generation
            new_mol = Chem.AddHs(mol)
            # Generate multiple 3D conformations
            res = AllChem.EmbedMultipleConfs(new_mol, numConfs=numConfs)
            # Optimize all conformations with MMFF force field
            res = AllChem.MMFFOptimizeMoleculeConfs(new_mol)
            # Select conformation with lowest energy
            index = np.argmin([x[1] for x in res])
            conf = new_mol.GetConformer(id=int(index))
            # Remove hydrogens for consistency with input
            new_mol = Chem.RemoveHs(new_mol)
        except Exception as e:
            # Fallback to 2D coordinates if 3D generation fails
            new_mol = mol
            AllChem.Compute2DCoords(new_mol)  # Generate 2D layout
            conf = new_mol.GetConformer()

        try:
            atom_poses = Compound3DKit.get_atom_poses(new_mol, conf)
        except ValueError as e:
            print(f"Error: {e}")  # Print any coordinate extraction errors

        return new_mol, atom_poses

    @staticmethod
    def get_bond_lengths(edges, atom_poses):
        """
        Calculate Euclidean distances between bonded atoms.

        Args:
            edges (array-like): Array of [source_atom, target_atom] pairs
            atom_poses (array-like): 3D coordinates for each atom

        Returns:
            numpy.ndarray: Array of bond lengths as float32 values

        Example:
            >>> bond_lengths = Compound3DKit.get_bond_lengths(edge_list, positions)
            >>> # Returns array of distances between bonded atoms
        """
        bond_lengths = []
        for src_node_i, tar_node_j in edges:
            # Calculate Euclidean distance between connected atoms
            bond_lengths.append(np.linalg.norm(atom_poses[tar_node_j] - atom_poses[src_node_i]))
        bond_lengths = np.array(bond_lengths, 'float32')
        return bond_lengths

    @staticmethod
    def get_superedge_angles(edges, atom_poses):
        """
        Calculate bond angles and create superedge graph for angular relationships.

        This method finds all pairs of bonds that share a common atom and calculates
        the angles between them, creating a higher-order graph representation.

        Args:
            edges (array-like): Array of bond connections [source, target]
            atom_poses (array-like): 3D atomic coordinates

        Returns:
            tuple: (super_edges, bond_angles, bond_angle_dirs)
                - super_edges: Connections between bonds that form angles
                - bond_angles: Calculated angles in radians
                - bond_angle_dirs: Direction indicators for the angles
        """

        def _get_vec(atom_poses, edge):
            """Helper function to get vector representation of a bond."""
            return atom_poses[edge[1]] - atom_poses[edge[0]]

        def _get_angle(vec1, vec2):
            """Calculate the angle between two vectors using dot product."""
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            if norm1 == 0 or norm2 == 0:
                return 0  # Return zero angle if either vector has zero length
            # Normalize vectors to unit length
            vec1 = vec1 / (norm1 + 1e-5)  # Add small epsilon to prevent division by zero
            vec2 = vec2 / (norm2 + 1e-5)
            # Calculate angle using dot product formula: cos(θ) = v1·v2
            angle = np.arccos(np.dot(vec1, vec2))
            return angle  # Return angle in radians

        EDGE = len(edges)  # Total number of edges in the molecular graph
        edge_indices = np.arange(EDGE)  # Create array of edge indices [0, 1, 2, ...]
        super_edges = []  # Will store pairs of edges that form angles
        bond_angles = []  # Will store calculated angles
        bond_angle_dirs = []  # Will store directionality information

        # Iterate through each edge as potential target edge
        for tar_edge_i in range(EDGE):
            tar_edge = edges[tar_edge_i]  # Current target edge [atom_i, atom_j]

            # Find all edges that share the same target atom (atom_i) as source
            # This creates angle relationships: src_edge -> shared_atom <- tar_edge
            src_edge_indices = edge_indices[edges[:, 1] == tar_edge[0]]

            for src_edge_i in src_edge_indices:
                if src_edge_i == tar_edge_i:
                    continue  # Skip self-comparison

                src_edge = edges[src_edge_i]  # Source edge forming angle with target
                src_vec = _get_vec(atom_poses, src_edge)  # Vector of source edge
                tar_vec = _get_vec(atom_poses, tar_edge)  # Vector of target edge

                super_edges.append([src_edge_i, tar_edge_i])  # Store edge pair
                angle = _get_angle(src_vec, tar_vec)  # Calculate angle between vectors
                bond_angles.append(angle)
                # Record whether edges share the same direction around central atom
                bond_angle_dirs.append(src_edge[1] == tar_edge[0])

        # Convert to numpy arrays or create empty arrays if no angles found
        if len(super_edges) == 0:
            super_edges = np.zeros([0, 2], 'int64')  # Empty superedge array
            bond_angles = np.zeros([0, ], 'float32')  # Empty angles array
        else:
            super_edges = np.array(super_edges, 'int64')
            bond_angles = np.array(bond_angles, 'float32')

        return super_edges, bond_angles, bond_angle_dirs


# ==============================================================================
# Graph Data Construction Functions
# ==============================================================================

def new_mol_to_graph_data(mol):
    """
    Convert an RDKit molecule to graph data format with comprehensive features.

    This function creates a graph representation of a molecule where atoms are nodes
    and bonds are edges. It includes both categorical and continuous features,
    and adds self-loops for complete graph connectivity.

    Args:
        mol (rdkit.Chem.Mol): RDKit molecule object to convert

    Returns:
        dict or None: Dictionary containing graph data with keys:
            - Atom features: 'atomic_num', 'degree', etc.
            - Bond features: 'bond_type', 'is_in_ring', etc.
            - 'edges': Array of [source, target] connections
            Returns None if molecule has no atoms.

    Example:
        >>> graph_data = new_mol_to_graph_data(mol)
        >>> # Returns dict with all atomic and bond features as arrays
    """
    # Check if molecule contains any atoms
    if len(mol.GetAtoms()) == 0:
        return None

    # Get feature names from CompoundKit vocabularies
    atom_id_names = list(CompoundKit.atom_vocab_dict.keys()) + CompoundKit.atom_float_names
    bond_id_names = list(CompoundKit.bond_vocab_dict.keys()) + CompoundKit.bond_float_names

    # Initialize data dictionary with empty lists for each atom feature
    data = {name: [] for name in atom_id_names}

    # Extract atom features for all atoms in molecule
    raw_atom_feat_dicts = CompoundKit.get_atom_names(mol)
    for atom_feat in raw_atom_feat_dicts:
        for name in atom_id_names:
            data[name].append(atom_feat[name])  # Collect features for each atom

    # Initialize bond feature lists and edge list
    for name in bond_id_names:
        data[name] = []
    data['edges'] = []

    # Process all bonds in the molecule
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()  # Source atom index
        j = bond.GetEndAtomIdx()  # Target atom index

        # Add edges in both directions for undirected graph representation
        data['edges'] += [(i, j), (j, i)]

        # Extract bond features and duplicate for both edge directions
        for name in bond_id_names:
            bond_feature_id = CompoundKit.get_bond_feature_id(bond, name)
            data[name] += [bond_feature_id] * 2  # Same features for both directions

    # Add self-loops for each atom (important for graph neural networks)
    N = len(data[atom_id_names[0]])  # Number of atoms
    for i in range(N):
        data['edges'] += [(i, i)]  # Self-loop: atom connects to itself

    # Add self-loop bond features (use last index in vocabulary as default)
    for name in bond_id_names:
        bond_feature_id = get_bond_feature_dims([name])[0] - 1  # Last vocab index
        data[name] += [bond_feature_id] * N  # One per atom

    # Convert all lists to appropriate numpy arrays
    # Categorical atom features -> int64
    for name in list(CompoundKit.atom_vocab_dict.keys()):
        data[name] = np.array(data[name], 'int64')
    # Continuous atom features -> float32
    for name in CompoundKit.atom_float_names:
        data[name] = np.array(data[name], 'float32')
    # Bond features -> int64
    for name in bond_id_names:
        data[name] = np.array(data[name], 'int64')
    # Edge connections -> int64
    data['edges'] = np.array(data['edges'], 'int64')

    return data


def new_smiles_to_graph_data(smiles, **kwargs):
    """
    Convert a SMILES string directly to graph data format.

    This is a convenience function that combines SMILES parsing with
    graph data conversion.

    Args:
        smiles (str): SMILES representation of the molecule
        **kwargs: Additional arguments (currently unused)

    Returns:
        dict or None: Graph data dictionary or None if SMILES is invalid

    Example:
        >>> graph_data = new_smiles_to_graph_data('CCO')  # ethanol
        >>> # Returns graph representation of ethanol
    """
    mol = AllChem.MolFromSmiles(smiles)  # Parse SMILES to RDKit molecule
    if mol is None:
        return None  # Invalid SMILES string
    data = new_mol_to_graph_data(mol)  # Convert to graph data
    return data


def mol_to_graph_data(mol):
    """
    Convert RDKit molecule to graph data with specific feature selection.

    This function creates a more focused graph representation using a specific
    subset of atomic and bond features, with different handling of feature indices.

    Args:
        mol (rdkit.Chem.Mol): RDKit molecule object

    Returns:
        dict or None: Graph data dictionary or None if conversion fails

    Note:
        This version adds +1 to atom feature indices for out-of-vocabulary handling
        and uses 0 as the out-of-vocabulary index for bond features.
    """
    # Validate input molecule
    if len(mol.GetAtoms()) == 0:
        return None

    # Define specific feature subsets (more focused than new_mol_to_graph_data)
    atom_id_names = [
        "atomic_num", "chiral_tag", "degree", "explicit_valence",
        "formal_charge", "hybridization", "implicit_valence",
        "is_aromatic", "total_numHs", 'num_radical_e',
        'atom_is_in_ring', 'valence_out_shell',
    ]
    bond_id_names = [
        "bond_dir", "bond_type", "is_in_ring", "is_conjugated", "bond_stereo",
    ]

    # Initialize data dictionary
    data = {name: [] for name in atom_id_names}
    data['van_der_waals_radis'] = []
    data['mass'] = []
    data.update({name: [] for name in bond_id_names})
    data['edges'] = []

    ### Extract atomic features with OOV handling
    for i, atom in enumerate(mol.GetAtoms()):
        if atom.GetAtomicNum() == 0:  # Skip dummy/invalid atoms
            return None
        # Add +1 to feature indices for out-of-vocabulary token at index 0
        for name in atom_id_names:
            data[name].append(CompoundKit.get_atom_feature_id(atom, name) + 1)
        # Add continuous atomic features
        data['van_der_waals_radis'].append(CompoundKit.get_atom_value(atom, 'van_der_waals_radis'))
        data['mass'].append(CompoundKit.get_atom_value(atom, 'mass'))

    ### Extract bond features
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        data['edges'] += [(i, j), (j, i)]  # Bidirectional edges

        for name in bond_id_names:
            # Use 0 as OOV index (no +1 offset like atoms)
            bond_feature_id = CompoundKit.get_bond_feature_id(bond, name)
            data[name] += [bond_feature_id] * 2

    ### Handle molecules with no bonds (rare edge case)
    if len(data['edges']) == 0:
        for name in bond_id_names:
            data[name] = np.zeros((0,), dtype="int64")
        data['edges'] = np.zeros((0, 2), dtype="int64")

    ### Convert to numpy arrays with appropriate data types
    for name in atom_id_names:
        data[name] = np.array(data[name], 'int64')
    data['mass'] = np.array(data['mass'], 'float32')
    data['van_der_waals_radis'] = np.array(data['van_der_waals_radis'], 'float32')
    for name in bond_id_names:
        data[name] = np.array(data[name], 'int64')
    data['edges'] = np.array(data['edges'], 'int64')

    return data


def mol_to_geognn_graph_data(mol, atom_poses, column_name):
    """
    Convert RDKit molecule and 3D coordinates to geometric graph data.

    This function creates an enhanced graph representation that includes
    3D geometric features like bond lengths and bond angles, suitable
    for geometric graph neural networks.

    Args:
        mol (rdkit.Chem.Mol): RDKit molecule object
        atom_poses (array-like): 3D coordinates for each atom [[x,y,z], ...]
        column_name (str): Name of chromatographic column (for compatibility)

    Returns:
        dict or None: Enhanced graph data with geometric features:
            - All standard graph features from mol_to_graph_data
            - 'atom_pos': 3D atomic coordinates
            - 'bond_length': Euclidean distances between bonded atoms
            - 'BondAngleGraph_edges': Connections between bonds forming angles
            - 'bond_angle': Angles between connected bonds

    Example:
        >>> geo_data = mol_to_geognn_graph_data(mol, positions, 'Cyclosil_B')
        >>> # Returns graph with geometric features for GeoGNN
    """
    if len(mol.GetAtoms()) == 0:
        return None

    # Get basic graph representation first
    data = mol_to_graph_data(mol)
    if data is None:
        return None

    # Add 3D geometric features
    data['atom_pos'] = np.array(atom_poses, 'float32')  # 3D coordinates
    data['bond_length'] = Compound3DKit.get_bond_lengths(data['edges'], data['atom_pos'])

    # Calculate bond angles and superedge relationships
    BondAngleGraph_edges, bond_angles, bond_angle_dirs = Compound3DKit.get_superedge_angles(
        data['edges'], data['atom_pos']
    )

    data['BondAngleGraph_edges'] = BondAngleGraph_edges  # Edge-to-edge connections
    data['bond_angle'] = np.array(bond_angles, 'float32')  # Angles in radians

    return data


def mol_to_geognn_graph_data_MMFF3d(mol, column_name):
    """
    Generate geometric graph data using MMFF-optimized 3D coordinates.

    This function combines 3D structure generation with geometric graph
    construction, providing a complete pipeline for creating 3D molecular graphs.

    Args:
        mol (rdkit.Chem.Mol): Input RDKit molecule
        column_name (str): Chromatographic column name (for compatibility)

    Returns:
        dict or None: Geometric graph data with MMFF-optimized coordinates

    Example:
        >>> geo_data = mol_to_geognn_graph_data_MMFF3d(mol, 'CP_Chirasil_D_Val')
        >>> # Returns graph with optimized 3D geometry
    """
    # Generate and optimize 3D molecular conformation
    mol, atom_poses = Compound3DKit.get_MMFF_atom_poses(mol, numConfs=10)
    # Convert to geometric graph data
    return mol_to_geognn_graph_data(mol, atom_poses, column_name)


# ==============================================================================
# 3D Molecular File I/O Functions
# ==============================================================================

def obtain_3D_mol(smiles, name):
    """
    Generate optimized 3D molecular structure and save to MOL file.

    This function takes a SMILES string, generates multiple 3D conformations,
    optimizes them using MMFF, and saves the best conformation to a file.

    Args:
        smiles (str): SMILES representation of the molecule
        name (str): Base filename for saving (without extension)

    Returns:
        rdkit.Chem.Mol: RDKit molecule with optimized 3D coordinates

    Raises:
        ValueError: If SMILES is invalid or 3D generation fails

    Example:
        >>> mol_3d = obtain_3D_mol('CCO', 'ethanol_3d')
        >>> # Creates 'ethanol_3d.mol' file with 3D coordinates
    """
    # Parse SMILES to molecule
    mol = AllChem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Unable to create molecule from smiles' {smiles} '")

    # Add hydrogens for realistic 3D structure
    new_mol = Chem.AddHs(mol)

    # Generate multiple conformations to find best geometry
    res = AllChem.EmbedMultipleConfs(new_mol, numConfs=10)
    if not res:
        raise ValueError(f"Unable to generate conformation for smiles' {smiles} '")

    # Optimize all conformations using MMFF force field
    res = AllChem.MMFFOptimizeMoleculeConfs(new_mol)

    # Remove hydrogens to match input format
    new_mol = Chem.RemoveHs(new_mol)

    # Save optimized structure to MOL file
    Chem.MolToMolFile(new_mol, name + '.mol')
    return new_mol


def save_3D_mol(all_smile, mol_save_dir):
    """
    Batch generate and save 3D molecular structures from SMILES list.

    This function processes a list of SMILES strings, generates 3D structures
    for each one, and saves them to individual MOL files. It tracks failures
    and continues processing even when some molecules fail.

    Args:
        all_smile (list): List of SMILES strings to process
        mol_save_dir (str): Directory to save 3D molecular structure files

    Returns:
        list: Indices of SMILES that failed to generate 3D structures

    Example:
        >>> failed_indices = save_3D_mol(smiles_list, 'structures_3d/')
        >>> print(f"Failed to generate 3D structures for {len(failed_indices)} molecules")
    """
    index = 0
    error_conformer = []  # Track failed molecule indices
    pbar = tqdm(all_smile)  # Progress bar for batch processing

    # Create output directory if it doesn't exist
    try:
        os.makedirs(f'{mol_save_dir}')
    except OSError:
        pass  # Directory already exists

    # Process each SMILES string
    for smiles in pbar:
        try:
            # Generate and save 3D structure
            obtain_3D_mol(smiles, f'{mol_save_dir}/3D_mol_{index}')
        except ValueError:
            # Record failure and continue with next molecule
            error_conformer.append(index)
            index += 1
            continue
        index += 1

    return error_conformer


def save_dataset(orderly_smile, mol_save_dir, orderly_name, descriptors_name, error_conformer, transfer_target):
    """
    Create and save complete dataset with both graph data and molecular descriptors.

    This function combines 3D molecular structures from MOL files with calculated
    molecular descriptors to create a comprehensive dataset for machine learning.
    It handles both chiral molecules and column molecules.

    Args:
        orderly_smile (list): Ordered list of SMILES strings
        mol_save_dir (str): Directory containing 3D molecular structure files
        orderly_name (str): Filename for saving graph dataset (without extension)
        descriptors_name (str): Filename for saving descriptors dataset (without extension)
        error_conformer (list): Indices of molecules that failed 3D generation
        transfer_target (str): Name of the chromatographic column

    Returns:
        list: Updated list of indices for molecules that failed processing

    Example:
        >>> final_errors = save_dataset(
        ...     smiles_list, 'structures/', 'graph_data', 'descriptors', 
        ...     initial_errors, 'Cyclosil_B'
        ... )
    """
    dataset = []  # Will store graph data for each molecule
    dataset_descriptors = []  # Will store descriptor vectors
    pbar = tqdm(orderly_smile)  # Progress tracking
    index = 0

    for Smiles in pbar:
        # Skip molecules that failed 3D generation
        if index in error_conformer:
            index += 1
            continue

        # Load chiral molecule from MOL file
        mol_Chiral_path = os.path.join(mol_save_dir, f"Chiral_3D_mol/3D_mol_{index}.mol")
        if not os.path.isfile(mol_Chiral_path):
            raise FileNotFoundError(f"File not found: {mol_Chiral_path}")

        mol_Chiral = Chem.MolFromMolFile(mol_Chiral_path)
        if mol_Chiral is None:
            raise ValueError(f"Invalid MOL file: {mol_Chiral_path}")

        # Load column molecule from MOL file
        mol_Column_path = os.path.join(mol_save_dir, f"3D_mol_0.mol")
        if not os.path.isfile(mol_Column_path):
            raise FileNotFoundError(f"File not found: {mol_Column_path}")

        mol_Column = Chem.MolFromMolFile(mol_Column_path)
        if mol_Column is None:
            raise ValueError(f"Invalid MOL file: {mol_Column_path}")

        # Calculate molecular descriptors for both chiral and column molecules
        descriptor_Chiral = calculate_descriptors(mol_Chiral, transfer_target, 'chiral')
        descriptor_Column = calculate_descriptors(mol_Column, transfer_target, 'column')

        # Combine descriptors from both molecules
        combined_descriptor = np.concatenate([descriptor_Chiral, descriptor_Column])
        dataset_descriptors.append(combined_descriptor)

        # Generate geometric graph data for the chiral molecule
        data = mol_to_geognn_graph_data_MMFF3d(mol_Chiral, transfer_target)
        if data is None:
            print(f"Warning: Unable to generate graph data for molecules indexed {index}. Skip.")
            error_conformer.append(index)
            index += 1
            continue

        dataset.append(data)
        index += 1

    # Convert to numpy array and save both datasets
    dataset_descriptors = np.array(dataset_descriptors)
    np.save(f"{orderly_name}.npy", dataset, allow_pickle=True)  # Graph data (needs pickle for complex objects)
    np.save(f'{descriptors_name}.npy', dataset_descriptors)  # Descriptor vectors (simple arrays)

    return error_conformer


# ==============================================================================
# Final Dataset Construction Function
# ==============================================================================

def Construct_dataset(dataset, RT, temperature_program, descriptor, column):
    """
    Construct a list of PyTorch Geometric Data objects from molecular dataset.

    Args:
        dataset: List of molecular data dictionaries containing atomic and bond features
        RT: List of retention times (target values) for each molecule
        temperature_program: List of temperature program values for each molecule
        descriptor: List of descriptor values for each molecule
        column: Column parameter

    Returns:
        graph: List of PyTorch Geometric Data objects ready for GAT training
    """
    graph = []

    # Process each molecule in the dataset
    for i in range(len(dataset)):
        data = dataset[i]
        tp = temperature_program[i]
        des = descriptor[i]

        # Initialize lists to store atomic and bond features
        atom_x = []
        atom_edge_attr = []

        # Extract atomic features based on predefined atom ID names
        for name in atom_id_names:
            atom_x.append(data[name])

        # Extract bond features based on predefined bond ID names
        for name in bond_id_names:
            atom_edge_attr.append(data[name])

        # Convert atomic features to PyTorch tensor (transpose to get correct shape)
        atom_x = torch.from_numpy(np.array(atom_x).T).to(torch.int64)

        # Add continuous atomic features (mass and van der Waals radius)
        atom_float_feature_mass = torch.from_numpy(data["mass"].astype(np.float32))
        atom_float_feature_van = torch.from_numpy(data["van_der_waals_radis"].astype(np.float32))
        atom_x = torch.cat([atom_x, atom_float_feature_mass.reshape(-1, 1)], dim=1)
        atom_x = torch.cat([atom_x, atom_float_feature_van.reshape(-1, 1)], dim=1)

        # Optional: Add atomic positions (currently commented out)
        # atom_pos = torch.from_numpy(data['atom_pos']).to(torch.float32)
        # atom_x = torch.cat([atom_x, atom_pos], dim=1)

        # Extract edge indices (connectivity information) and transpose for PyG format
        atom_edge_index = torch.from_numpy(data['edges'].T).to(torch.int64)

        # Convert bond features to PyTorch tensor (transpose to get correct shape)
        atom_edge_attr = torch.from_numpy(np.array(atom_edge_attr).T).to(torch.int64)

        # Add bond length as continuous edge feature
        bond_float_feature = torch.from_numpy(data['bond_length'].astype(np.float32))
        atom_edge_attr = torch.cat([atom_edge_attr, bond_float_feature.reshape(-1, 1)], dim=1)

        # Add descriptor information to each edge
        des = torch.tensor([des], dtype=torch.float32)
        des_expanded = des.expand(atom_edge_attr.size(0), -1)  # Broadcast to all edges
        atom_edge_attr = torch.cat((atom_edge_attr, des_expanded), dim=1)

        # Add temperature program information to each edge
        tp = torch.tensor([tp], dtype=torch.float32)
        tp_expanded = tp.expand(atom_edge_attr.size(0), -1)  # Broadcast to all edges
        atom_edge_attr = torch.cat((atom_edge_attr, tp_expanded), dim=1)

        # Set target value (retention time)
        y = torch.tensor([RT[i]], dtype=torch.float32)

        # Create PyTorch Geometric Data object
        data = Data(x=atom_x, edge_index=atom_edge_index, edge_attr=atom_edge_attr, y=y)

        # Add to graph list
        graph.append(data)

    return graph