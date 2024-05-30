import pandas as pd
import anndata

class ImmunoPhenoError(Exception):
    """A base class for ImmunoPheno Exceptions."""

class PlotUMAPError(ImmunoPhenoError):
    """No normalized counts are found when plotting a normalized UMAP"""

class ImmunoPhenoData:
    def __init__(self,
                 protein_matrix: str | pd.DataFrame = None,
                 gene_matrix: str | pd.DataFrame = None,
                 cell_labels: str | pd.DataFrame = None,
                 spreadsheet: str = None,
                 scanpy: anndata.AnnData = None,
                 scanpy_labels: str = None):

        # Raw values
        self._protein_matrix = protein_matrix
        self._gene_matrix = gene_matrix
        self._spreadsheet = spreadsheet
        self._cell_labels = cell_labels
        self._scanpy = scanpy
        self._label_certainties = None

        # Temp values (for resetting index)
        self._temp_protein = None
        self._temp_gene = None
        self._temp_labels = None
        self._temp_certainties = None

    def reset_index(self, arg1: int | float = None, arg2: int | float = None):
        """Resets the index for dataframes and adds/subtracts a value.

        Resets the index and then adds a value if needed. The value for "arg1"
        will be added, while the value for "arg2" will be subtracted.

        Args:
            arg1 (int | float): the first number who can have a second
                line if it needed it
            arg2 (int | float): the second number

        Returns:
            None. Modifies the protein dataset in-place.
        """
        self._protein_matrix = self._temp_protein + arg1 - arg2
            