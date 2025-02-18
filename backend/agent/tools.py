import requests
from typing import List, Union, Tuple, Dict
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Draw
from rdkit.Chem import DataStructs


def evaluate_expression(expression:str) -> int:
    """evaluates any numerical expression and gives numerical result

    Args:
        expression: a string representing a Python expression 
    """
    return eval(expression)

def get_smiles_from_name(molecule_name: str) -> str:
    """Get SMILES notation for a molecule using its common name via PubChem API.

    Args:
        molecule_name: Common name of the molecule

    Raises:
        ValueError: If molecule name is not found or API request fails
    """
    try:
        # Step 1: Search for the compound by name to get PubChem CID
        search_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{molecule_name}/cids/JSON"
        response = requests.get(search_url)
        if response.status_code != 200:
            raise ValueError(f"Molecule '{molecule_name}' not found in PubChem")
        
        cid = response.json()['IdentifierList']['CID'][0]
        
        # Step 2: Get SMILES using the CID
        smiles_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/IsomericSMILES/JSON"
        response = requests.get(smiles_url)
        if response.status_code != 200:
            raise ValueError(f"Failed to retrieve SMILES for CID {cid}")
        
        smiles = response.json()['PropertyTable']['Properties'][0]['IsomericSMILES']
        return smiles
        
    except requests.exceptions.RequestException as e:
        raise ValueError(f"Network error while accessing PubChem: {str(e)}")
    except (KeyError, IndexError) as e:
        raise ValueError(f"Error parsing PubChem response: {str(e)}")
    except Exception as e:
        raise ValueError(f"Unknown error: {str(e)}")

def _validate_smiles(smiles: str) -> Chem.Mol:
    """Validate SMILES and convert to RDKit Mol object.

    Args:
        smiles: SMILES string representation of the molecule

    Raises:
        ValueError: If SMILES parsing fails
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")
    return mol

def calculate_molecular_descriptors(smiles: str) -> Dict[str, float]:
    """Calculate important molecular descriptors for a given molecule. Results contains MolecularWeight,LogP,NumRotatableBonds,NumHAcceptors,NumHDonors,TPSA,NumRings,NumAromatic

    Args:
        smiles: SMILES string representation of the molecule
    """
    mol = _validate_smiles(smiles)
    descriptors = {
        'MolecularWeight': Descriptors.ExactMolWt(mol),
        'LogP': Descriptors.MolLogP(mol),
        'NumRotatableBonds': Descriptors.NumRotatableBonds(mol),
        'NumHAcceptors': Descriptors.NumHAcceptors(mol),
        'NumHDonors': Descriptors.NumHDonors(mol),
        'TPSA': Descriptors.TPSA(mol),
        'NumRings': Descriptors.RingCount(mol),
        'NumAromatic': len([ring for ring in mol.GetRingInfo().AtomRings() 
                          if all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring)])
    }
    return descriptors

def check_lipinski_rules(smiles: str) -> Dict[str, bool]:
    """Check Lipinski's Rule of Five compliance for drug-likeness.

    Args:
        smiles: SMILES string representation of the molecule
    """
    mol = _validate_smiles(smiles)
    mw = Descriptors.ExactMolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hbd = Descriptors.NumHDonors(mol)
    hba = Descriptors.NumHAcceptors(mol)
    
    rules = {
        'MW_under_500': mw <= 500,
        'LogP_under_5': logp <= 5,
        'HBD_under_5': hbd <= 5,
        'HBA_under_10': hba <= 10,
        'Passes_All_Rules': all([mw <= 500, logp <= 5, hbd <= 5, hba <= 10])
    }
    return rules

def calculate_fingerprint(smiles: str, fp_type: str = "morgan") -> List[int]:
    """Generate molecular fingerprints of various types.

    Args:
        smiles: SMILES string representation of the molecule
        fp_type: Type of fingerprint ('morgan', 'maccs', 'topological')
    """
    mol = _validate_smiles(smiles)
    if fp_type == "morgan":
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    elif fp_type == "maccs":
        fp = AllChem.GetMACCSKeysFingerprint(mol)
    elif fp_type == "topological":
        fp = Chem.RDKFingerprint(mol)
    else:
        raise ValueError(f"Unsupported fingerprint type: {fp_type}")
        
    return list(fp.ToBitString())

def calculate_similarity(smiles1: str, smiles2: str, similarity_metric: str = "tanimoto") -> float:
    """Calculate similarity between two molecules using various metrics.

    Args:
        smiles1: SMILES string of first molecule
        smiles2: SMILES string of second molecule
        similarity_metric: Metric to use ('tanimoto', 'dice', 'cosine')
    """
    mol1 = _validate_smiles(smiles1)
    mol2 = _validate_smiles(smiles2)
    
    fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, nBits=2048)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, nBits=2048)
    
    if similarity_metric == "tanimoto":
        return DataStructs.TanimotoSimilarity(fp1, fp2)
    elif similarity_metric == "dice":
        return DataStructs.DiceSimilarity(fp1, fp2)
    elif similarity_metric == "cosine":
        return DataStructs.CosineSimilarity(fp1, fp2)
    else:
        raise ValueError(f"Unsupported similarity metric: {similarity_metric}")

def find_substructure_matches(smiles: str, substructure_smarts: str) -> List[List[int]]:
    """Find all matches of a substructure pattern in a molecule.

    Args:
        smiles: SMILES string representation of the molecule
        substructure_smarts: SMARTS pattern to search for
    """
    mol = _validate_smiles(smiles)
    pattern = Chem.MolFromSmarts(substructure_smarts)
    if pattern is None:
        raise ValueError("Invalid SMARTS pattern")
    matches = mol.GetSubstructMatches(pattern)
    return [list(match) for match in matches]

def convert_3d_coordinates(smiles: str) -> List[List[float]]:
    """Generate 3D coordinates for a molecule.

    Args:
        smiles: SMILES string representation of the molecule
    """
    mol = _validate_smiles(smiles)
    mol = Chem.AddHs(mol)
    try:
        AllChem.EmbedMolecule(mol, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(mol)
        conf = mol.GetConformer()
        coordinates = []
        for i in range(mol.GetNumAtoms()):
            pos = conf.GetAtomPosition(i)
            coordinates.append([pos.x, pos.y, pos.z])
        return coordinates
    except:
        raise ValueError("Failed to generate 3D coordinates")

def predict_reactions(smiles: str) -> List[str]:
    """Predict possible reactions for a given molecule using SMARTS reaction templates.

    Args:
        smiles: SMILES string of the reactant
    """
    mol = _validate_smiles(smiles)
    
    # Example reaction SMARTS patterns (can be expanded)
    reaction_templates = [
        '[C:1](=[O:2])-[OH:3]>>[C:1](=[O:2])[O-].[H+]',  # Deprotonation
        '[C:1](=[O:2])-[OH:3]>>[C:1](=[O:2])[NH2]'       # Amidation
    ]
    
    products = []
    for template in reaction_templates:
        rxn = AllChem.ReactionFromSmarts(template)
        products_tuples = rxn.RunReactants((mol,))
        
        for product_tuple in products_tuples:
            for product in product_tuple:
                products.append(Chem.MolToSmiles(product))
                
    return list(set(products))  # Remove duplicates

def analyze_scaffold(smiles: str) -> Dict[str, Union[str, int]]:
    """Analyze the molecular scaffold and return key information about the core structure.

    Args:
        smiles: SMILES string representation of the molecule
    """
    mol = _validate_smiles(smiles)
    from rdkit.Chem.Scaffolds import MurckoScaffold
    
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    framework = MurckoScaffold.MakeScaffoldGeneric(scaffold)
    
    analysis = {
        'scaffold_smiles': Chem.MolToSmiles(scaffold),
        'generic_scaffold': Chem.MolToSmiles(framework),
        'num_rings_scaffold': Descriptors.RingCount(scaffold),
        'num_aromatic_scaffold': len([ring for ring in scaffold.GetRingInfo().AtomRings() 
                                    if all(scaffold.GetAtomWithIdx(i).GetIsAromatic() 
                                          for i in ring)])
    }
    return analysis

def get_molecular_formula(smiles: str) -> str:
    """Get the molecular formula for a given molecule.

    Args:
        smiles: SMILES string representation of the molecule
    """
    mol = _validate_smiles(smiles)
    mol = Chem.AddHs(mol)  # Include implicit hydrogens
    
    atom_dict = {}
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        atom_dict[symbol] = atom_dict.get(symbol, 0) + 1
    
    # Sort atoms in standard order (C, H, then alphabetically)
    ordered_atoms = []
    if 'C' in atom_dict:
        ordered_atoms.append(('C', atom_dict.pop('C')))
    if 'H' in atom_dict:
        ordered_atoms.append(('H', atom_dict.pop('H')))
    ordered_atoms.extend(sorted(atom_dict.items()))
    
    formula = ''
    for symbol, count in ordered_atoms:
        formula += symbol
        if count > 1:
            formula += str(count)
            
    return formula


# Add your new chemical analysis tools here
CHEM_TOOLS = [
    get_smiles_from_name,
    calculate_molecular_descriptors,
    check_lipinski_rules,
    calculate_fingerprint,
    calculate_similarity,
    find_substructure_matches,
    convert_3d_coordinates,
    predict_reactions,
    analyze_scaffold,
    get_molecular_formula
]