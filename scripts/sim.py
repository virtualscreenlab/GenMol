import os
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import MACCSkeys
from rdkit import DataStructs
from typing import List, Dict, Tuple
from collections import defaultdict


class MACCSKeySimilarityAnalyzer:
    """
    Analyze molecular similarity using MACCS (Molecular ACCess System) keys.
    MACCS keys are 166-bit structural fingerprints where each bit indicates
    presence/absence of a predefined substructure pattern [citation:5][citation:10].
    """

    def __init__(self, reference_smiles: str = None):
        self.reference_smiles = reference_smiles or "CCS(=O)(=O)N1CC(CC#N)C1"
        self.reference_fp = self._compute_maccs_fingerprint(self.reference_smiles)

    def _compute_maccs_fingerprint(self, smiles: str):
        """
        Generate MACCS fingerprint for a given SMILES string.
        Returns a 167-bit vector (RDKit implementation includes 1 extra bit).

        Note: MACCS keys are 166-bit but RDKit returns 167 bits due to
        0-indexing convention [citation:5][citation:10].
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"Warning: Invalid SMILES string: {smiles}")
            return None
        return MACCSkeys.GenMACCSKeys(mol)

    def _compute_similarity(self, fp1, fp2, metric: str = "tanimoto") -> float:
        """
        Calculate similarity between two MACCS fingerprints.

        Args:
            fp1, fp2: MACCS fingerprint objects
            metric: similarity metric ('tanimoto' or 'dice')

        Returns:
            Similarity score between 0 and 1
        """
        if fp1 is None or fp2 is None:
            return 0.0

        if metric.lower() == "tanimoto":
            return DataStructs.TanimotoSimilarity(fp1, fp2)
        elif metric.lower() == "dice":
            return DataStructs.DiceSimilarity(fp1, fp2)
        else:
            raise ValueError(f"Unsupported metric: {metric}. Use 'tanimoto' or 'dice'.")

    def load_databases(self, db_paths: Dict[str, str]) -> Dict[str, List[str]]:
        """Load SMILES strings from database files."""
        databases = {}
        for db_name, file_path in db_paths.items():
            try:
                with open(file_path, 'r') as f:
                    smiles_list = [line.strip() for line in f if line.strip()]
                databases[db_name] = smiles_list
                print(f"✓ Loaded {len(smiles_list)} ligands from {db_name}")
            except FileNotFoundError:
                print(f"⚠ File {file_path} not found. Using sample data for {db_name}.")
                databases[db_name] = self._get_sample_ligands()
        return databases

    def _get_sample_ligands(self) -> List[str]:
        """Sample ligand data for demonstration."""
        return [
            "CCS(=O)(=O)N1CC(CC#N)(c2ccc(C(F)(F)F)cc2)C1",
            "CCS(=O)(=O)N1CC(CC#N)(c2cccc(C(F)(F)F)c2)C1",
            "CCS(=O)(=O)N1CC(CC#N)(NC(=O)CCC(F)(F)F)C1",
            "CCS(=O)(=O)N1CC(CC#N)(OC(=O)c2ccccc2F)C1",
            "CCS(=O)(=O)N1CC(CC#N)(NC(=O)CCC(C)(F)F)C1",
            "CCS(=O)(=O)N1CC(CC#N)(NC(=O)CCC2CC2)C1",
            "CCS(=O)(=O)N1CC(CC#N)(OC(=O)CCC(F)(F)F)C1",
            "CCS(=O)(=O)N1CC(CC#N)(OC(=O)CCC(C)(F)F)C1",
            "CCS(=O)(=O)N1CC(CC#N)(OC(=O)CCC2CC2)C1",
            "CCS(=O)(=O)N1CC(CC#N)(OC(=O)CC2=CCCCC2)C1",
            "CCS(=O)(=O)N1CC(CC#N)(C(=O)Cc2ccc(Br)s2)C1",
            "S(=O)(=O)(N1CC(C1)CC#N)CC(C(F)(F)CC)(C)C",
            "S(=O)(=O)(N1CC(C[NH3+])C1)CC"
        ]

    def find_intersection(self, databases: Dict[str, List[str]]) -> List[str]:
        """Find ligands present in ALL databases."""
        if len(databases) < 3:
            raise ValueError("Need at least 3 databases for intersection analysis")

        # Convert each list to set and find intersection
        db_sets = [set(smiles_list) for smiles_list in databases.values()]
        intersection = set.intersection(*db_sets)

        print(f"\n✓ Found {len(intersection)} ligands present in all databases")
        return list(intersection)

    def calculate_maccs_bit_frequency(self, ligands: List[str]) -> Dict[int, float]:
        """
        Analyze which MACCS key bits are most common in the ligand set.
        This helps understand structural features prevalent in your compounds.
        """
        bit_counts = defaultdict(int)
        n_valid = 0

        for smiles in ligands:
            fp = self._compute_maccs_fingerprint(smiles)
            if fp is not None:
                n_valid += 1
                # Convert to list and count which bits are set (value = 1)
                bit_list = list(fp)
                for i, is_set in enumerate(bit_list):
                    if is_set:
                        bit_counts[i] += 1

        # Calculate frequency
        frequencies = {bit: count / n_valid for bit, count in bit_counts.items()}
        return frequencies

    def rank_ligands(self, ligands: List[str],
                     similarity_metric: str = "tanimoto") -> List[Tuple[str, float]]:
        """
        Rank ligands by similarity to reference compound.

        Args:
            ligands: List of SMILES strings
            similarity_metric: 'tanimoto' or 'dice'

        Returns:
            List of (smiles, similarity_score) sorted descending
        """
        results = []

        for smiles in ligands:
            fp = self._compute_maccs_fingerprint(smiles)
            score = self._compute_similarity(self.reference_fp, fp, similarity_metric)
            results.append((smiles, score))

        # Sort by score descending
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def compute_similarity_matrix(self, ligands: List[str]) -> np.ndarray:
        """
        Compute pairwise similarity matrix for all ligands.
        Useful for clustering analysis [citation:1].
        """
        n = len(ligands)
        matrix = np.zeros((n, n))

        # Generate all fingerprints
        fps = [self._compute_maccs_fingerprint(smiles) for smiles in ligands]

        # Calculate pairwise similarities
        for i in range(n):
            for j in range(i, n):
                if fps[i] is not None and fps[j] is not None:
                    sim = self._compute_similarity(fps[i], fps[j])
                    matrix[i, j] = sim
                    matrix[j, i] = sim

        return matrix

    def cluster_by_similarity(self, similarity_matrix: np.ndarray,
                              threshold: float = 0.6) -> List[List[int]]:
        """
        Simple greedy clustering based on similarity threshold.
        For hierarchical clustering, use scipy.cluster.hierarchy [citation:1].
        """
        n = len(similarity_matrix)
        visited = [False] * n
        clusters = []

        for i in range(n):
            if not visited[i]:
                cluster = [i]
                visited[i] = True
                for j in range(i + 1, n):
                    if not visited[j] and similarity_matrix[i, j] >= threshold:
                        cluster.append(j)
                        visited[j] = True
                clusters.append(cluster)

        return clusters

    def print_analysis(self, results: Dict):
        """Pretty print the analysis results."""
        print("\n" + "=" * 80)
        print("MACCS KEYS SIMILARITY ANALYSIS")
        print("=" * 80)
        print(f"Reference compound: {self.reference_smiles}")
        print(f"MACCS fingerprint size: {len(self.reference_fp)} bits (167 with RDKit indexing)")

        print("\n" + "-" * 80)
        print(f"IDENTICAL LIGANDS IN ALL DATABASES: {results['total_identical']}")
        print("-" * 80)

        for i, smiles in enumerate(results['identical_ligands'], 1):
            print(f"{i:2d}. {smiles[:70]}..." if len(smiles) > 70 else f"{i:2d}. {smiles}")

        print("\n" + "-" * 80)
        print("TOP 5 HIGHEST SCORING (most similar to reference)")
        print("-" * 80)
        for i, (smiles, score) in enumerate(results['top_5'], 1):
            print(f"{i}. Score: {score:.4f}  |  {smiles[:60]}...")

        print("\n" + "-" * 80)
        print("BOTTOM 5 LOWEST SCORING (least similar to reference)")
        print("-" * 80)
        for i, (smiles, score) in enumerate(results['bottom_5'], 1):
            print(f"{i}. Score: {score:.4f}  |  {smiles[:60]}...")

        print("\n" + "-" * 80)
        print("SUMMARY STATISTICS")
        print("-" * 80)
        for db_name, size in results['database_sizes'].items():
            print(f"  {db_name}: {size} ligands")
        print(f"  Identical ligands in all databases: {results['total_identical']}")
        print(f"  Average MACCS similarity score: {results['avg_score']:.4f}")
        print(f"  Score range: {results['min_score']:.4f} - {results['max_score']:.4f}")
        print(f"  Standard deviation: {results['std_score']:.4f}")

        # Most common MACCS key bits (structural features)
        if results.get('bit_frequencies'):
            print("\n" + "-" * 80)
            print("MOST COMMON STRUCTURAL FEATURES (MACCS bits present in >50% of ligands)")
            print("-" * 80)
            common_bits = [(bit, freq) for bit, freq in results['bit_frequencies'].items()
                           if freq > 0.5]
            common_bits.sort(key=lambda x: x[1], reverse=True)
            for bit, freq in common_bits[:10]:
                print(f"  Bit {bit:3d}: present in {freq * 100:.1f}% of ligands")

    def export_results(self, results: Dict, output_file: str = "maccs_similarity_results.csv"):
        """Export results to CSV file."""
        if not results or not results.get('ranked_ligands'):
            print("No results to export.")
            return

        df = pd.DataFrame(results['ranked_ligands'],
                          columns=['SMILES', 'MACCS_Similarity'])
        df['Rank'] = range(1, len(df) + 1)

        # Add molecular formula (optional)
        df['Valid_Molecule'] = df['SMILES'].apply(
            lambda x: Chem.MolFromSmiles(x) is not None
        )

        # Reorder columns
        df = df[['Rank', 'SMILES', 'MACCS_Similarity', 'Valid_Molecule']]

        df.to_csv(output_file, index=False)
        print(f"\n✓ Results exported to {output_file}")

        return df

    def run_complete_analysis(self,
                              database_paths: Dict[str, str],
                              similarity_metric: str = "tanimoto") -> Dict:
        """
        Run complete analysis pipeline.

        Args:
            database_paths: Dict with keys 'db1', 'db2', 'db3' and file paths
            similarity_metric: 'tanimoto' or 'dice'

        Returns:
            Dictionary with all analysis results
        """
        # Load databases
        databases = self.load_databases(database_paths)

        # Find identical ligands
        identical = self.find_intersection(databases)

        if not identical:
            print("\n⚠ No identical ligands found across all databases.")
            return {}

        # Rank by similarity
        ranked = self.rank_ligands(identical, similarity_metric)
        scores = [score for _, score in ranked]

        # Calculate MACCS bit frequencies
        bit_frequencies = self.calculate_maccs_bit_frequency(identical)

        # Compute similarity matrix for clustering
        sim_matrix = self.compute_similarity_matrix(identical)

        # Perform clustering
        clusters = self.cluster_by_similarity(sim_matrix, threshold=0.6)

        results = {
            'identical_ligands': identical,
            'ranked_ligands': ranked,
            'top_5': ranked[:5],
            'bottom_5': ranked[-5:],
            'total_identical': len(identical),
            'avg_score': np.mean(scores),
            'min_score': np.min(scores),
            'max_score': np.max(scores),
            'std_score': np.std(scores),
            'database_sizes': {name: len(smiles) for name, smiles in databases.items()},
            'bit_frequencies': bit_frequencies,
            'similarity_matrix': sim_matrix,
            'clusters': clusters,
            'similarity_metric': similarity_metric
        }

        return results


def main():
    """Main execution function."""

    # Define database file paths (UPDATE THESE PATHS!)
    database_paths = {
        'db1': 'C:/Users/PC1/PycharmProjects/pythonProject1/files/Genmol/database1.smi',
        'db2': 'C:/Users/PC1/PycharmProjects/pythonProject1/files/Genmol/database2.smi',
        'db3': 'C:/Users/PC1/PycharmProjects/pythonProject1/files/Genmol/database3.smi'
    }

    # Initialize analyzer
    print("Initializing MACCS Keys Similarity Analyzer...")
    analyzer = MACCSKeySimilarityAnalyzer()

    # Run analysis with Tanimoto similarity (standard for MACCS)
    print("\n" + "=" * 80)
    print("RUNNING MACCS KEYS ANALYSIS")
    print("=" * 80)
    results = analyzer.run_complete_analysis(database_paths, similarity_metric="tanimoto")

    if results:
        # Print formatted results
        analyzer.print_analysis(results)

        # Export to CSV
        analyzer.export_results(results, "maccs_ligand_analysis.csv")

        # Optional: Display cluster information
        print("\n" + "-" * 80)
        print(f"CLUSTERING RESULTS (threshold = 0.6)")
        print("-" * 80)
        print(f"Found {len(results['clusters'])} clusters among {results['total_identical']} ligands")
        for i, cluster in enumerate(results['clusters']):
            if len(cluster) > 1:
                print(f"  Cluster {i + 1}: {len(cluster)} ligands")
    else:
        print("\n⚠ No identical ligands found. Consider checking your database files.")

    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()