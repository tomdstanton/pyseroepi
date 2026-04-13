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
    """
    Container for sparse distance matrices between isolates.

    This class provides a memory-efficient way to store and manipulate pairwise
    distances, supporting various metric types and conversions.

    Attributes:
        matrix (csr_array): The underlying sparse distance matrix in CSR format.
        index (pd.Series): Sample identifiers corresponding to the matrix rows/columns.
        metric_type (MetricType): The type of metric represented (e.g., LITERAL_DISTANCE).
        max_value (float): The maximum possible value for the metric, used for
            conversions between literal and normalized types.

    Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>> from pyseroepi.dist import Distances, MetricType
        >>> q = pd.Series(['S1', 'S1'])
        >>> t = pd.Series(['S2', 'S3'])
        >>> w = pd.Series([5, 10])
        >>> dists = Distances.from_pairwise(q, t, w)
        >>> print(dists.matrix.toarray())
    """
    matrix: csr_array
    index: pd.Series
    metric_type: MetricType
    max_value: float = None  # Required if converting between literal and normalised

    def __post_init__(self):
        """Validates the consistency of the distance matrix and labels."""
        # Airtight check 1: Dimensions must match
        if self.matrix.shape[0] != self.matrix.shape[1]:
            raise ValueError("Distance matrix must be square.")
        if len(self.index) != self.matrix.shape[0]:
            raise ValueError("Number of labels must match matrix dimensions.")
        self.index.name = 'sample_id'

    @classmethod
    def from_pairwise(cls, query_col: pd.Series, target_col: pd.Series, weight_col: pd.Series,
                      metric_type: MetricType = MetricType.LITERAL_DISTANCE) -> Self:
        """
        Creates a Distances instance from long-format pairwise data.

        Args:
            query_col: Series containing the first isolate IDs.
            target_col: Series containing the second isolate IDs.
            weight_col: Series containing the distances/similarities.
            metric_type: The type of metric provided. Defaults to LITERAL_DISTANCE.

        Returns:
            A new Distances instance.
        """
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
        """
        Parses a pairwise distance matrix from SKA2 output.

        Args:
            filepath_or_buffer: Path to the SKA2 distance file.

        Returns:
            A new Distances instance.
        """
        df = pd.read_table(filepath_or_buffer, usecols=(0, 1, 2))
        return cls.from_pairwise(df.iloc[:, 0], df.iloc[:, 1], df.iloc[:, 2])

    @classmethod
    def from_pathogenwatch(cls, filepath_or_buffer) -> Self:
        """
        Parses a square distance matrix from Pathogenwatch.

        Args:
            filepath_or_buffer: Path to the Pathogenwatch CSV file.

        Returns:
            A new Distances instance.
        """
        df = pd.read_csv(filepath_or_buffer, index_col=0)
        M = coo_array(df.values)
        M = M.maximum(M.T)
        return cls(M, df.columns, MetricType.LITERAL_DISTANCE)

    @classmethod
    def from_newick(cls, newick_string: str) -> Self:
        """
        Parses a Newick string and calculates patristic distances.

        Requires Biopython (`pip install biopython`).

        Args:
            newick_string: The Newick tree string.

        Returns:
            A new Distances instance.

        Raises:
            ImportError: If biopython is not installed.
        """
        try:
            from Bio import Phylo
        except ImportError:
            raise ImportError("biopython is required to calculate patristics. Install with pyseroepi[dev]")

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
            index=pd.Series(labels),
            metric_type=MetricType.LITERAL_DISTANCE
        )

    def connected_components(self, threshold: int = 20) -> pd.Series:
        """
        Identifies connected components (clusters) based on a distance threshold.

        Args:
            threshold: Maximum distance to consider isolates as connected.
                Defaults to 20 (e.g., 20 SNPs).

        Returns:
            A Series of cluster labels indexed by isolate IDs.
        """
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

        return pd.Series(labels, index=self.index, dtype='category', name=f"connected_components_{threshold=}")

    def apply_clustering(self, clusterer, **kwargs) -> pd.Series:
        """
        Applies an external clustering algorithm to the distance matrix.

        Args:
            clusterer: An object implementing `fit_predict` (e.g., sklearn.cluster.DBSCAN).
            **kwargs: Additional arguments passed to `fit_predict`.

        Returns:
            A Series of cluster labels.

        Raises:
            TypeError: If the clusterer does not implement `fit_predict`.
        """
        if not hasattr(clusterer, "fit_predict"):
            raise TypeError("The provided model must implement a 'fit_predict' method.")

        # We ensure the metric is a distance, not a similarity, for standard clusterers
        distance_obj = self.to_type(MetricType.NORMALISED_DISTANCE)

        # Call the external method on our raw sparse array
        cluster_labels = clusterer.fit_predict(distance_obj.matrix, **kwargs)

        # Return a pandas Series mapped to our original labels
        return pd.Series(cluster_labels, index=self.index, name=f"{clusterer}_cluster")

    def to_type(self, target_type: MetricType) -> Self:
        """
        Converts the distances to a different metric type.

        Args:
            target_type: The desired target MetricType.

        Returns:
            A new Distances instance with the converted matrix.

        Raises:
            ValueError: If conversion requires `max_value` but it is not set.
        """
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
