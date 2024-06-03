import pandas as pd
import anndata

class ImmunoPhenoError(Exception):
    """A base class for ImmunoPheno Exceptions."""

class PlotUMAPError(ImmunoPhenoError):
    """No normalized counts are found when plotting a normalized UMAP"""

class ImmunoPhenoData:
    """A class to hold single-cell data (CITE-Seq, etc) and cytometry data.
    
    Performs fitting of gaussian/negative binomial mixture models and
    normalization to antibodies present in a protein dataset. 

    Args:
        protein_matrix (str | pd.Dataframe): file path or dataframe to ADT count matrix
            Where: Row index (cells) x column (antibodies)
        gene_matrix (str | pd.DataFrame): file path or dataframe to UMI count matrix
            Where: Row index (cells) x column (genes)
        cell_labels (str | pd.DataFrame): file path or dataframe
            Where: Row index (cells) x column (cell type such as Cell Ontology ID)
        spreadsheet (str): name of csv file containing a spreadsheet with
            information about the experiment and antibodies. Used for uploading to a database
        scanpy (anndata.AnnData): scanpy anndata object used to load in protein and gene data
        scanpy_labels (str): name of the field inside a scanpy object with the cell labels
            Where: scanpy is an AnnData object containing an 'obs' field
                Ex: AnnData.obs['scanpy_labels']
    """

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
        # Calculated values
        self._all_fits = None
        self._all_fits_dict = None
        self._cumulative = False
        self._last_normalize_params = None
        self._raw_umap = None
        self._norm_umap = None
        self._umap_kwargs = None
        self._normalized_counts_df = None
        self._classified_filt_df = None
        self._cell_labels_filt_df = None
        self._linear_reg_df = None
        self._z_scores_df = None
        self._singleR_rna = None
        self._ab_ids_dict = None

        # Used when sending data to the server for running STvEA
        self._background_cell_z_score = -10
        self._stvea_correction_value = 0

        # If loading in a scanpy object
        if scanpy is not None:
            # Extract and load protein data
            protein_anndata = scanpy[:, scanpy.var["feature_types"] == "Antibody Capture"].copy()
            protein_df = protein_anndata.to_df(layer="counts")
            self._protein_matrix = protein_df.copy(deep=True)
            self._temp_protein = self._protein_matrix.copy(deep=True)

            # Extract and load rna/gene data
            rna_anndata = scanpy[:, scanpy.var["feature_types"] == "Gene Expression"].copy()
            gene_df = rna_anndata.to_df(layer="counts")
            self._gene_matrix = gene_df.copy(deep=True)
            self._temp_gene = self._gene_matrix.copy(deep=True)

            # Filter out rna based on genes used for SingleR
            self._singleR_rna = _singleR_rna(self._gene_matrix)

            # Use scanpy cell labels if present
            if scanpy_labels is not None:
                try:
                    labels = scanpy.obs[scanpy_labels]
                    # Load these labels into the class
                    self._cell_labels = pd.DataFrame(labels)
                    self._temp_labels = self._cell_labels.copy(deep=True)
                except:
                    raise Exception("Field not found in scanpy object")

        if protein_matrix is None and scanpy is None:
            raise LoadMatrixError("protein_matrix file path or dataframe must be provided")

        if (protein_matrix is not None and
            cell_labels is not None and
            gene_matrix is None and
            scanpy is None):
            raise LoadMatrixError("gene_matrix file path or dataframe must be present along with "
                                  "cell_labels")

        # Single cell
        if self._protein_matrix is not None and self._gene_matrix is not None and scanpy is None:
            self._protein_matrix = _load_adt(self._protein_matrix) # assume user provides cells as rows, ab as col
            self._temp_protein = self._protein_matrix.copy(deep=True)

            self._gene_matrix = _load_rna(self._gene_matrix) # assume user provides cells as rows, genes as col
            self._singleR_rna = _singleR_rna(self._gene_matrix)
            self._temp_gene = self._gene_matrix.copy(deep=True)

        # Flow
        elif self._protein_matrix is not None and self._gene_matrix is None and scanpy is None:
            self._protein_matrix = _load_adt(self._protein_matrix)  # assume user provides cells as rows, ab as col
            self._temp_protein = self._protein_matrix.copy(deep=True)
            self._gene_matrix = None

        # If dealing with single cell data with provided cell labels
        if self._cell_labels is not None:
            self._cell_labels = _load_labels(self._cell_labels) # assume user provides cells as rows, label as col
            self._cell_labels_filt_df = self._cell_labels.copy(deep=True) # if loading in labels intiailly, also place them in the norm labels
            self._temp_labels = self._cell_labels.copy(deep=True)
        else:
            cell_labels = None

        # If filtering antibodies using a provided spreadsheet for database uploads
        if spreadsheet is not None:
            self._protein_matrix = _filter_antibodies(self._protein_matrix, spreadsheet)
            self._temp_protein = self._protein_matrix.copy(deep=True)

            # Also create a dictionary of antibodies with their IDs for name conversion
            self._ab_ids_dict = _target_ab_dict(_read_antibodies(spreadsheet))

    def __getitem__(self, index: pd.Index | list):
        """Allows instances of ImmunoPhenoData to use the indexing operator.

        Args:
            index (pd.Index | list): list or pandas index of cell names. This will return
                a new ImmunoPhenoData object containing only those cell names in 
                all dataframes of the object. 

        Returns:
            ImmunoPhenoData: new instance containing modified dataframes such as
                protein, RNA, labels, etc.
        """
        if isinstance(index, list):
            index = pd.Index(index)

        new_instance = ImmunoPhenoData(self._protein_matrix)  # Create a new instance of the class
        
        for attr_name, attr_value in self.__dict__.items():
            if isinstance(attr_value, pd.DataFrame):
                setattr(new_instance, attr_name, attr_value.loc[index] if attr_value is not None else None)
            elif isinstance(attr_value, anndata.AnnData):
                obs_index = [i for i, obs in enumerate(attr_value.obs_names) if obs in index]
                setattr(new_instance, attr_name, attr_value[obs_index] if attr_value is not None else None)
            else:
                setattr(new_instance, attr_name, attr_value)
        
        return new_instance

    @property
    def protein(self) -> pd.DataFrame:
        """Get or set the current protein dataframe in the object.
        
        Setting a new protein dataframe requires the format to have rows (cells) and
        columns (proteins/antibodies). 

        Returns:
            pd.DataFrame: dataframe containing protein data.
        """
        return self._protein_matrix

    @protein.setter
    def protein(self, value: pd.DataFrame) -> None:
        self._protein_matrix = value

    @property
    def rna(self) -> pd.DataFrame:
        """Get or set the current gene/rna dataframe in the object.
        
        Setting a new RNA dataframe requires the format to have rows (cells) and
        columns (genes).

        Returns:
            pd.DataFrame: dataframe containing RNA data.
        """
        return self._gene_matrix

    @rna.setter
    def rna(self, value: pd.DataFrame) -> None:
        self._gene_matrix = value
    
    @property
    def fits(self) -> dict:
        """Get the mixture model fits for each antibody in the protein dataframe.

        Each mixture model fit will be stored in a dictionary, where the key
        is the name of the antibody. 

        Returns:
            dict: key-value pairs represent an antibody name with a
            nested dictionary containing the respective mixture model fits. 
            Fits are ranked by the lowest AIC.
        """
        return self._all_fits_dict

    @property
    def normalized_counts(self) -> pd.DataFrame:
        """Get the normalized protein dataframe in the object.

        This dataframe will only be present if normalize_all_antibodies 
        has been called. The format will have rows (cells) and columns (proteins/antibodies).
        Note that some rows may be missing/filtered out from the normalization step.

        Returns:
            pd.DataFrame: normalized protein counts for each antibody. 
        """
        return self._normalized_counts_df
    
    @property
    def labels(self) -> pd.DataFrame:
        """Get or set the current cell labels for all normalized cells in the object.

        This dataframe will contain rows (cells) and two columns: "labels" and "celltypes". 
        All values in the "labels" column will follow the EMBL-EBI Cell Ontology ID format.
        A common name for each value in "labels" will be in the "celltypes" column.

        Setting a new cell labels table will only update rows (cells) that are shared 
        between the existing and new table. 

        Returns:
            pd.DataFrame: dataframe containing two columns: "labels" and "celltypes". 
        """
        return self._cell_labels_filt_df
    
    @labels.setter
    def labels(self, value: pd.DataFrame) -> None:
        self._cell_labels_filt_df = value #  Change the norm_cell_labels
        # If the cells in 'value' are found in the original table, update those rows too
        common_indices = self._cell_labels.index.intersection(self._cell_labels_filt_df.index)
        # Check for missing rows in new table that should be updated in original labels
        missing_indices = self._cell_labels_filt_df.index.difference(self._cell_labels.index)

        if not missing_indices.empty:
            print(f"Warning: The following rows were not found in the original protein dataset and will be ignored: {missing_indices.tolist()}")

        # Check for indices in raw_cell_labels that will be updated
        if not common_indices.empty:
            # Reset the UMAPs
            self._raw_umap = None
            self._norm_umap = None
            try:
                # Update the rows in the raw_cell_labels to reflect the annotations in the norm_cell_labels
                self._cell_labels.loc[common_indices, ['labels', 'celltype']] = self._cell_labels_filt_df.loc[common_indices, ['labels', 'celltype']]
            except Exception as e:
                print(f"An error occurred during the update: {e}")
        else:
            print("No common rows found between old and new labels. No updates will be made to the old labels.")

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
            