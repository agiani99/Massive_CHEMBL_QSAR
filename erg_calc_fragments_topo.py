import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, Crippen, Fragments
from rdkit.Chem import rdmolops
import uuid
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class FixedPharmacophoreAnalyzer:
    """
    Pharmacophore analyzer for ErG fingerprint generation with RDKit 2D fragment descriptors
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = bool(verbose)
        self.feature_definitions = {
            'HA': 'Hydrogen Bond Acceptor',
            'HD': 'Hydrogen Bond Donor', 
            'AR': 'Aromatic Ring Centroid',
            'HY': 'Hydrophobic Centroid',
            'PLUS': 'Positively Charged Atom [+]',
            'MINUS': 'Negatively Charged Atom [-]',
            'FF': 'Flip/Flop [#7,#8][H]'
        }
        
        # Get all fragment descriptor functions (fr_*)
        self.fragment_descriptors = self._get_fragment_descriptors()
        if self.verbose:
            print(f"Found {len(self.fragment_descriptors)} fragment descriptors (fr_*)")

    def _get_fragment_descriptors(self):
        """Get all fr_* descriptors from RDKit"""
        fragment_funcs = {}
        
        # Get all attributes from Fragments module that start with 'fr_'
        for name in dir(Fragments):
            if name.startswith('fr_'):
                func = getattr(Fragments, name)
                if callable(func):
                    fragment_funcs[name] = func
        
        return fragment_funcs
    
    def calculate_fragment_descriptors(self, mol):
        """Calculate all fr_* fragment descriptors"""
        fragment_values = {}
        
        for name, func in self.fragment_descriptors.items():
            try:
                value = func(mol)
                fragment_values[name] = value
            except Exception as e:
                # If calculation fails, set to NaN
                fragment_values[name] = np.nan
        
        return fragment_values

    def identify_hydrophobic_groups(self, mol):
        """Identify HY groups as connected components of non-hetero substituted non-aromatic carbons (2D only)"""
        hydrophobic_centroids = {}
        
        carbon_atoms = []
        for atom in mol.GetAtoms():
            if (atom.GetSymbol() == 'C' and 
                not atom.GetIsAromatic() and
                atom.GetFormalCharge() == 0):
                
                neighbors = [neighbor.GetSymbol() for neighbor in atom.GetNeighbors()]
                # Works with implicit Hs: neighbors will be only heavy atoms
                if all(symbol in ['C', 'H'] for symbol in neighbors):
                    carbon_atoms.append(atom.GetIdx())
        
        if len(carbon_atoms) >= 3:
            carbon_groups = self._find_connected_carbon_groups(mol, carbon_atoms)
            centroid_idx = -2000
            for group in carbon_groups:
                if len(group) >= 3:
                    hydrophobic_centroids[centroid_idx] = {
                        'atoms': list(group),
                        'type': 'HY'
                    }
                    centroid_idx -= 1
        
        return hydrophobic_centroids
    
    def _find_connected_carbon_groups(self, mol, carbon_atoms):
        """Find connected components of carbon atoms"""
        carbon_set = set(carbon_atoms)
        visited = set()
        groups = []
        
        for atom_idx in carbon_atoms:
            if atom_idx not in visited:
                group = []
                stack = [atom_idx]
                
                while stack:
                    current = stack.pop()
                    if current not in visited:
                        visited.add(current)
                        group.append(current)
                        
                        atom = mol.GetAtomWithIdx(current)
                        for neighbor in atom.GetNeighbors():
                            neighbor_idx = neighbor.GetIdx()
                            if neighbor_idx in carbon_set and neighbor_idx not in visited:
                                stack.append(neighbor_idx)
                
                if len(group) >= 3:
                    groups.append(group)
        
        return groups
    
    def identify_pharmacophore_features(self, mol):
        """Hierarchical identification of pharmacophore features"""
        features = {}
        centroids = {}
        
        # Step 1: Aromatic rings
        aromatic_rings, ar_centroids = self.identify_aromatic_rings(mol)
        centroids.update(ar_centroids)
        for centroid_idx in ar_centroids:
            features[centroid_idx] = 'AR'
        
        # Step 2: Hydrophobic groups
        hy_centroids = self.identify_hydrophobic_groups(mol)
        centroids.update(hy_centroids)
        for centroid_idx in hy_centroids:
            features[centroid_idx] = 'HY'
        
        # Step 3: Charged atoms
        plus_atoms, minus_atoms = self.identify_ionizable_features(mol)
        for atom_idx in plus_atoms:
            features[atom_idx] = 'PLUS'
        for atom_idx in minus_atoms:
            features[atom_idx] = 'MINUS'
        
        # Step 4: Flip-flop atoms
        ff_atoms = self.identify_flip_flop_atoms(mol)
        self.ff_atoms = ff_atoms
        
        # Step 5: HA and HD atoms
        assigned_atoms = set(atom_idx for atom_idx in features.keys() if atom_idx >= 0)
        
        ha_atoms = self.identify_ha_atoms(mol, exclude=assigned_atoms)
        for atom_idx in ha_atoms:
            features[atom_idx] = 'HA'
        
        assigned_atoms.update(ha_atoms)
        hd_atoms = self.identify_hd_atoms(mol, exclude=assigned_atoms)
        for atom_idx in hd_atoms:
            features[atom_idx] = 'HD'
               
        return features, centroids, aromatic_rings
              
    def identify_ionizable_features(self, mol):
        """Identify PLUS and MINUS charged atoms"""
        plus_atoms = set()
        minus_atoms = set()
        
        for atom in mol.GetAtoms():
            atom_idx = atom.GetIdx()
            formal_charge = atom.GetFormalCharge()
            
            if formal_charge > 0:
                plus_atoms.add(atom_idx)
            elif formal_charge < 0:
                minus_atoms.add(atom_idx)
        
        return plus_atoms, minus_atoms

    def identify_flip_flop_atoms(self, mol):
        """Identify FF atoms using SMARTS [#7,#8][H]"""
        ff_pattern = Chem.MolFromSmarts('[#7,#8][H]')
        ff_atoms = set()
        
        if ff_pattern:
            matches = mol.GetSubstructMatches(ff_pattern)
            for match in matches:
                ff_atoms.add(match[0])
        
        return ff_atoms
        
    def identify_ha_atoms(self, mol, exclude=None):
        """Identify hydrogen bond acceptor atoms"""
        if exclude is None:
            exclude = set()
            
        ha_atoms = set()
        for atom in mol.GetAtoms():
            atom_idx = atom.GetIdx()
            if atom_idx in exclude:
                continue
                
            symbol = atom.GetSymbol()
            
            if symbol == 'O':
                ha_atoms.add(atom_idx)
            elif symbol == 'N':
                if atom.GetFormalCharge() <= 0:
                    ha_atoms.add(atom_idx)
            elif symbol in ['S', 'F', 'Cl', 'Br']:
                ha_atoms.add(atom_idx)
                
        return ha_atoms
        
    def identify_aromatic_rings(self, mol):
        """Identify aromatic rings and create centroids (2D only, no coordinates)"""
        aromatic_rings = []
        aromatic_centroids = {}
        
        ri = mol.GetRingInfo()
        ring_idx = 0
        
        for ring in ri.AtomRings():
            if (len(ring) in [5, 6, 7] and 
                all(mol.GetAtomWithIdx(idx).GetIsAromatic() for idx in ring)):
                
                aromatic_rings.append(ring)
                centroid_idx = -(ring_idx + 1000)
                aromatic_centroids[centroid_idx] = {
                    'atoms': list(ring),
                    'type': 'AR',
                    'ring_size': len(ring),
                    'ring_id': ring_idx
                }
                ring_idx += 1
        
        return aromatic_rings, aromatic_centroids
        
    def identify_hd_atoms(self, mol, exclude=None):
        """Identify hydrogen bond donor atoms (no explicit Hs required)"""
        if exclude is None:
            exclude = set()
        hd_atoms = set()
        for atom in mol.GetAtoms():
            idx = atom.GetIdx()
            if idx in exclude:
                continue
            symbol = atom.GetSymbol()
            if symbol in ['O', 'N', 'S']:
                # Use implicit/explicit H count; do not require explicit H neighbors
                if atom.GetTotalNumHs() > 0:
                    hd_atoms.add(idx)
        return hd_atoms
    
    def get_feature_counts_for_fingerprint(self, features):
        """Get feature counts for ErG fingerprint"""
        counts = {'HA': 0, 'HD': 0, 'AR': 0, 'HY': 0, 'PLUS': 0, 'MINUS': 0}
        
        for idx, feature in features.items():
            if feature in counts:
                counts[feature] += 1
        
        if hasattr(self, 'ff_atoms') and self.ff_atoms:
            for atom_idx in self.ff_atoms:
                if atom_idx in features and features[atom_idx] == 'HA':
                    counts['HD'] += 1
        
        return counts
    
    def calculate_topological_distance(self, mol, idx1, idx2, centroids, dist_matrix):
        """
        Shortest-path (bond-count) distance between features.
        For AR/HY centroids use the minimal path among their member atoms.
        """
        def atoms_from_feature_index(i):
            if i < 0 and centroids and i in centroids:
                return centroids[i]['atoms']
            return [i]

        atoms1 = atoms_from_feature_index(idx1)
        atoms2 = atoms_from_feature_index(idx2)

        min_d = None
        for a in atoms1:
            for b in atoms2:
                if a == b:
                    continue
                d = int(dist_matrix[a, b])
                if min_d is None or d < min_d:
                    min_d = d

        if min_d is None:
            return None
        return max(1, min_d)  # clamp to at least 1

    def calculate_ph4_pairs(self, features, mol, centroids):
        """Calculate pharmacophore pairs using topological (bond-count) distances"""
        pairs = []
        feature_indices = list(features.keys())
        # Precompute topological distance matrix on the 2D graph (no Hs required)
        dist_matrix = rdmolops.GetDistanceMatrix(mol)

        # Base feature-feature pairs (unique unordered)
        for i, idx1 in enumerate(feature_indices):
            for idx2 in feature_indices[i+1:]:
                f1 = features[idx1]
                f2 = features[idx2]
                d = self.calculate_topological_distance(mol, idx1, idx2, centroids, dist_matrix)
                if d is None:
                    continue
                pairs.append({'feature1': f1, 'feature2': f2, 'distance': d})

        # Flip-flop: [#7,#8][H] atoms act as HD toward other features (no HA–HD self-pair at d=0)
        if hasattr(self, 'ff_atoms') and self.ff_atoms:
            for idx in feature_indices:
                if idx in self.ff_atoms and features.get(idx) == 'HA':
                    for other_idx in feature_indices:
                        if other_idx == idx:
                            continue
                        f2 = features[other_idx]
                        d = self.calculate_topological_distance(mol, idx, other_idx, centroids, dist_matrix)
                        if d is None:
                            continue
                        pairs.append({'feature1': 'HD', 'feature2': f2, 'distance': d})

        return pairs

    def generate_erg_fingerprint(self, mol, smiles_string=""):
        """
        Generate ErG fingerprint with topological distance bins (1..15) + RDKit fragment descriptors.
        Adds fuzzy spillover (0.3) to adjacent bins (d-1 and d+1). No 3D embedding.
        """
        try:
            # 2D fragment descriptors
            fragment_desc = self.calculate_fragment_descriptors(mol)

            # Identify features
            features, centroids, aromatic_rings = self.identify_pharmacophore_features(mol)
            pairs = self.calculate_ph4_pairs(features, mol, centroids)
            
            # Feature counts
            feature_counts = self.get_feature_counts_for_fingerprint(features)

            # Pair types
            feature_types = ['HA', 'HD', 'AR', 'HY', 'PLUS', 'MINUS']
            all_pair_types = []
            for i, f1 in enumerate(feature_types):
                for f2 in feature_types[i:]:
                    all_pair_types.append(f"{f1}_{f2}")
            
            # Initialize distance features (d1..d15)
            distance_features = {}
            for pair_type in all_pair_types:
                for d in range(1, 16):
                    distance_features[f"{pair_type}_d{d}"] = 0.0

            # Count shortest-path distances with fuzzy spillover (±1 bins get +0.3)
            fuzzy = 0.3
            for p in pairs:
                f1, f2 = p['feature1'], p['feature2']
                d = max(1, min(15, int(p['distance'])))
                pair_type = f"{f1}_{f2}" if f1 <= f2 else f"{f2}_{f1}"
                base_key = f"{pair_type}_d{d}"
                if base_key in distance_features:
                    distance_features[base_key] += 1.0
                # d-1
                if d > 1:
                    minus_key = f"{pair_type}_d{d-1}"
                    if minus_key in distance_features:
                        distance_features[minus_key] += fuzzy
                # d+1
                if d < 15:
                    plus_key = f"{pair_type}_d{d+1}"
                    if plus_key in distance_features:
                        distance_features[plus_key] += fuzzy

            # Basic properties (heavy-atom graph)
            mol_no_h = Chem.RemoveHs(mol)
            mw = rdMolDescriptors.CalcExactMolWt(mol_no_h)
            logp = rdMolDescriptors.CalcCrippenDescriptors(mol_no_h)[0]
            hbd = rdMolDescriptors.CalcNumHBD(mol_no_h)
            hba = rdMolDescriptors.CalcNumHBA(mol_no_h)

            fingerprint = {
                'SMILES': smiles_string,
                'MW': round(mw, 2),
                'LogP': round(logp, 2),
                'HBD': hbd,
                'HBA': hba
            }

            # Add feature counts
            for feat_type, count in feature_counts.items():
                fingerprint[f'count_{feat_type}'] = count

            # Add distance features
            fingerprint.update(distance_features)

            # Add fragment descriptors
            fingerprint.update(fragment_desc)

            return fingerprint

        except Exception as e:
            if getattr(self, "verbose", False):
                print(f"Error generating fingerprint: {str(e)[:100]}")
            return None


def process_csv_with_erg_descriptors(input_csv, output_csv):
    """
    Process CSV file and add ErG descriptors + RDKit fragment descriptors for each SMILES (2D/topological)
    """
    print(f"Loading CSV file: {input_csv}")
    df = pd.read_csv(input_csv)
    print(f"Found {len(df)} rows")

    if 'SMILES' not in df.columns:
        print("Error: 'SMILES' column not found in CSV!")
        return
    
    analyzer = FixedPharmacophoreAnalyzer()
    erg_fingerprints = []
    failed_smiles = []
    error_types = {'invalid_smiles': 0, 'fingerprint_generation': 0, 'empty_smiles': 0}
    
    print("\nProcessing SMILES strings...")
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Calculating ErG + fragment descriptors"):
        smiles = row['SMILES']
        if pd.isna(smiles) or smiles == '':
            failed_smiles.append((idx, 'Empty SMILES'))
            error_types['empty_smiles'] += 1
            erg_fingerprints.append(None)
            continue

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            failed_smiles.append((idx, 'Invalid SMILES'))
            error_types['invalid_smiles'] += 1
            erg_fingerprints.append(None)
            continue

        # No explicit hydrogens, no 3D
        fingerprint = analyzer.generate_erg_fingerprint(mol, smiles)
        if fingerprint is None:
            failed_smiles.append((idx, 'Fingerprint generation failed'))
            error_types['fingerprint_generation'] += 1
            erg_fingerprints.append(None)
        else:
            erg_fingerprints.append(fingerprint)
    
    print("\nCreating fingerprint dataframe...")
    erg_df = pd.DataFrame([fp for fp in erg_fingerprints if fp is not None])
    if 'SMILES' in erg_df.columns:
        erg_df = erg_df.drop('SMILES', axis=1)
    
    success_indices = [i for i, fp in enumerate(erg_fingerprints) if fp is not None]
    for col in erg_df.columns:
        df.loc[success_indices, col] = erg_df[col].values
    
    print(f"\nSaving results to: {output_csv}")
    df.to_csv(output_csv, index=False)
    
    print("\n" + "="*60)
    print("PROCESSING SUMMARY")
    print("="*60)
    print(f"Total rows processed: {len(df)}")
    print(f"Successfully processed: {len(erg_df)}")
    print(f"Failed: {len(failed_smiles)}")
    print(f"Total descriptor columns added: {len(erg_df.columns)}")
    print(f"  - ErG descriptors (MW, LogP, HBD, HBA, counts, distances): {len([c for c in erg_df.columns if not c.startswith('fr_')])}")
    print(f"  - Fragment descriptors (fr_*): {len([c for c in erg_df.columns if c.startswith('fr_')])}")

    if error_types:
        print(f"\nError breakdown:")
        print(f"  Empty SMILES: {error_types['empty_smiles']}")
        print(f"  Invalid SMILES: {error_types['invalid_smiles']}")
        print(f"  Fingerprint generation failed: {error_types['fingerprint_generation']}")
    
    if failed_smiles:
        print(f"\nFailed SMILES (first 10):")
        for idx, reason in failed_smiles[:10]:
            smiles_preview = str(df.iloc[idx]['SMILES'])[:60] if pd.notna(df.iloc[idx]['SMILES']) else 'N/A'
            print(f"  Row {idx}: {reason} - SMILES: {smiles_preview}...")
    
    print(f"\nOutput saved to: {output_csv}")
    print("="*60)

if __name__ == "__main__":
    import argparse, os, sys
    parser = argparse.ArgumentParser(description="Generate 2D/topological ErG + RDKit fragment descriptors from a CSV with a SMILES column.")
    parser.add_argument("input_csv", help="Input CSV path (must contain a 'SMILES' column)")
    parser.add_argument("-o", "--output", dest="output_csv", default=None, help="Output CSV path (defaults to <input>_topo.csv)")
    args = parser.parse_args()

    in_csv = args.input_csv
    if not os.path.isfile(in_csv):
        print(f"Error: input file not found: {in_csv}")
        sys.exit(1)

    if args.output_csv:
        out_csv = args.output_csv
    else:
        base, ext = os.path.splitext(in_csv)
        out_csv = f"{base}_topo{ext or '.csv'}"

    process_csv_with_erg_descriptors(in_csv, out_csv)