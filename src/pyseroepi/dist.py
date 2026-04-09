from dataclasses import dataclass, replace
from enum import Enum, auto
from typing import Self
from io import StringIO
import pandas as pd
import numpy as np
from scipy.sparse import coo_array, csr_array
from scipy.sparse.csgraph import connected_components as sp_connected_components


# Constants ------------------------------------------------------------------------------------------------------------
class MetricType(Enum):
    LITERAL_DISTANCE = auto()  # e.g., 5 SNPs
    NORMALISED_DISTANCE = auto()  # e.g., 0.05 Hamming
    LITERAL_SIMILARITY = auto()  # e.g., 95 shared nucleotides
    NORMALISED_SIMILARITY = auto()  # e.g., 0.95 Jaccard


# Classes --------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class Distances:
    matrix: csr_array
    labels: pd.Series
    metric_type: MetricType
    max_value: float = None  # Required if converting between literal and normalised

    def __post_init__(self):
        # Airtight check 1: Dimensions must match
        if self.matrix.shape[0] != self.matrix.shape[1]:
            raise ValueError("Distance matrix must be square.")
        if len(self.labels) != self.matrix.shape[0]:
            raise ValueError("Number of labels must match matrix dimensions.")

    @classmethod
    def from_pairwise(cls, query_col: pd.Series, target_col: pd.Series, weight_col: pd.Series,
                      metric_type: MetricType = MetricType.LITERAL_DISTANCE) -> Self:
        # 1. Drop to pure NumPy for speed
        q_vals = query_col.values
        t_vals = target_col.values
        # 2. Concatenate the two arrays into one long 1D array
        all_nodes = np.concatenate([q_vals, t_vals])
        # 3. np.unique with return_inverse gives us the unique labels AND
        # instantly maps every original string to its new integer index!
        uids, mapped_indices = np.unique(all_nodes, return_inverse=True)
        # 4. Split the mapped indices back into rows and columns
        half = len(q_vals)
        rows = mapped_indices[:half]
        cols = mapped_indices[half:]
        # 5. Build the matrix
        n = len(uids)
        M = coo_array((weight_col.values, (rows, cols)), shape=(n, n))
        # 6. Symmetrize and ensure it returns a CSR matrix
        return cls(M.maximum(M.T), pd.Series(uids), metric_type)

    @classmethod
    def from_ska2(cls, filepath_or_buffer) -> Self:
        """A pairwise, tab delimited distance matrix from SKA2 distance command"""
        df = pd.read_table(filepath_or_buffer, usecols=(0, 1, 2))
        return cls.from_pairwise(df.iloc[:, 0], df.iloc[:, 1], df.iloc[:, 2])

    # @classmethod
    # def from_mash(cls, filepath_or_buffer) -> Self:
    #     df = pd.read_table(filepath, usecols=(0, 1, 3))

    # @classmethod
    # def from_ska1(cls, filepath_or_buffer) -> Self:
    #     df = pd.read_table(filepath_or_buffer, usecols=(0, 1, 6))

    @classmethod
    def from_pathogenwatch(cls, filepath_or_buffer) -> Self:
        """A square, comma-delimited file representing SNP-distances between core genes"""
        df = pd.read_csv(filepath_or_buffer, index_col=0)
        M = coo_array(df.values)
        M = M.maximum(M.T)
        return cls(M, df.columns, MetricType.LITERAL_DISTANCE)

    @classmethod
    def from_newick(cls, newick_string: str) -> Self:
        """
        Parses a Newick string using Biopython and calculates patristic distances.
        Requires Biopython (`pip install biopython`).
        """
        try:
            from Bio import Phylo
        except ImportError as e:
            raise ImportError("Biopython must be installed to calculate patristics (pip install biopython)") from e

        # 1. Parse the tree
        tree = Phylo.read(StringIO(newick_string), "newick")

        # 2. Extract terminals (leaves)
        terminals = tree.get_terminals()
        labels = [leaf.name for leaf in terminals]
        n = len(terminals)

        # 3. Populate a dense NumPy array (symmetric distance calculation)
        matrix = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(i + 1, n):
                dist = tree.distance(terminals[i], terminals[j])
                matrix[i, j] = dist
                matrix[j, i] = dist

                # 4. Convert to CSR array and return as a Distances instance
        return cls(
            matrix=csr_array(matrix),
            labels=pd.Series(labels),
            metric_type=MetricType.LITERAL_DISTANCE
        )

    def connected_components(self, threshold: int = 20) -> pd.Series:
        # 1. Make a copy of the CSR array to avoid mutating the frozen original
        adj = self.matrix.copy()

        # 2. Convert to a binary adjacency array:
        # If distance <= threshold, make it a 1 (Valid Edge).
        # If distance > threshold, make it a 0 (Severed Edge).
        adj.data = np.where(adj.data <= threshold, 1, 0)

        # 3. Safely eliminate the 0s (which are now only the severed edges).
        # Your identical clones are safe because their distance of 0 was turned into a 1!
        adj.eliminate_zeros()

        # 4. Ensure every sample is connected to itself on the diagonal
        adj.setdiag(1)

        # 5. Run the clustering algorithm
        _, labels = sp_connected_components(csgraph=adj, directed=False, return_labels=True)

        return pd.Series(labels, index=self.labels.values, dtype='str', name="Cluster")

    def apply_clustering(self, clusterer, **kwargs) -> pd.Series:
        """
        Applies an external clustering algorithm (like sklearn's DBSCAN)
        to the underlying distance matrix.

        The clusterer MUST implement a `fit_predict` method and accept
        'precomputed' distance matrices if it relies on distances.
        """
        if not hasattr(clusterer, "fit_predict"):
            raise TypeError("The provided model must implement a 'fit_predict' method.")

        # We ensure the metric is a distance, not a similarity, for standard clusterers
        distance_obj = self.to_type(MetricType.NORMALISED_DISTANCE)

        # Call the external method on our raw sparse array
        cluster_labels = clusterer.fit_predict(distance_obj.matrix, **kwargs)

        # Return a pandas Series mapped to our original labels
        return pd.Series(cluster_labels, index=self.labels.values, name="Cluster")

    def to_type(self, target_type: MetricType) -> Self:
        """Returns a new Distances instance converted to the target metric type."""
        if self.metric_type == target_type:
            return self

        # If crossing the Literal <-> Normalised boundary, we need max_value
        needs_max = {MetricType.LITERAL_DISTANCE, MetricType.LITERAL_SIMILARITY}
        targets_norm = {MetricType.NORMALISED_DISTANCE, MetricType.NORMALISED_SIMILARITY}

        if (self.metric_type in needs_max and target_type in targets_norm) or \
                (self.metric_type in targets_norm and target_type in needs_max):
            if self.max_value is None:
                raise ValueError(f"Cannot convert between Literal and Normalised without a max_value.")

        # 1. Standardize to Normalised Distance first (as a base state)
        if self.metric_type == MetricType.LITERAL_DISTANCE:
            base_mat = self.matrix / self.max_value
        elif self.metric_type == MetricType.LITERAL_SIMILARITY:
            base_mat = 1.0 - (self.matrix / self.max_value)
        elif self.metric_type == MetricType.NORMALISED_SIMILARITY:
            base_mat = 1.0 - self.matrix
        else:
            base_mat = self.matrix

        # 2. Convert from base state (Normalised Distance) to Target
        if target_type == MetricType.NORMALISED_DISTANCE:
            new_mat = base_mat
        elif target_type == MetricType.NORMALISED_SIMILARITY:
            new_mat = 1.0 - base_mat
        elif target_type == MetricType.LITERAL_DISTANCE:
            new_mat = base_mat * self.max_value
        elif target_type == MetricType.LITERAL_SIMILARITY:
            new_mat = (1.0 - base_mat) * self.max_value

        # Return a new frozen instance
        return replace(self, matrix=new_mat, metric_type=target_type)
