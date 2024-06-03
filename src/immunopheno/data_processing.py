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
        protein_matrix (str | pd.Dataframe): file path or dataframe to ADT count matrix. 
            Format: Row (cells) x column (proteins/antibodies).
        gene_matrix (str | pd.DataFrame): file path or dataframe to UMI count matrix.
            Format: Row (cells) x column (genes).
        cell_labels (str | pd.DataFrame): file path or dataframe to cell type labels. 
            Format: Row (cells) x column (cell type such as Cell Ontology ID). The column
            name should be called "labels". 
        spreadsheet (str): name of csv file containing a spreadsheet with
            information about the experiment and antibodies. Used for uploading data to a database.
        scanpy (anndata.AnnData): scanpy AnnData object used to load in protein and gene data.
        scanpy_labels (str): location of cell labels inside a scanpy object. 
            Format: scanpy is an AnnData object containing an 'obs' field
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
            ImmunoPhenoData: contains modified dataframes based on provided rows/cells names
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

    def convert_labels(self) -> None:
        """Convert all cell ontology IDs to a common name.

        Requires all values in the "labels" column of the cell labels dataframe to
        follow the cell ontology format of CL:XXXXXXX or CL_XXXXXXX,
        where an "X" is a numeric value.

        Raises:
            Exception: The object does not contain cell labels 
            Exception: The cell labels dataframe does not contain a "labels" column
        """
        # First, check that the raw cell types table exists
        if self._cell_labels is not None and isinstance(self._cell_labels, pd.DataFrame):
            # Check if the "labels" column exists
            if "labels" in self._cell_labels:
                # Create mapping dictionary using values in the "labels" field
                labels_map = _ebi_idCL_map(self._cell_labels)

                # Map all values from dictionary back onto the "celltype" field
                temp_df = self._cell_labels.copy(deep=True)
                temp_df['celltype'] = temp_df['labels'].map(labels_map)

                # Ensure all "labels" follow the format of "CL:XXXXXXX"
                temp_df["labels"] = temp_df["labels"].str.replace(r'^CL_([0-9]+)$', r'CL:\1')
                
                # Set new table
                self._cell_labels = temp_df

                # Check if normalized cell types exist. If so, repeat above
                if self.labels is not None and isinstance(self.labels, pd.DataFrame):
                    norm_temp_df = self.labels.copy(deep=True)
                    norm_temp_df['celltype'] = norm_temp_df['labels'].map(labels_map)

                    # Ensure all "labels" follow the format of "CL:XXXXXXX"
                    norm_temp_df["labels"] = norm_temp_df["labels"].str.replace(r'^CL_([0-9]+)$', r'CL:\1')

                    # Set new table
                    self._cell_labels_filt_df = norm_temp_df
            else:
                raise Exception("Table does not contain 'labels' column")
        else:
            raise Exception("No cell labels found. Please provide a table with a 'labels' column.")

    def remove_antibody(self, antibody: str) -> None:
        """Removes an antibody from the protein data and mixture model fits

        Removes all values for an antibody from the protein dataframe in-place. If
        fit_antibody or fit_all_antibodies has been called, it will also remove 
        the mixture model fits for that antibody.

        Args:
            antibody (str): name of antibody to be removed

        Raises:
            AntibodyLookupError: The antibody is not found in the protein data
            AntibodyLookupError: The provided antibody is not a string
        """
        # CHECK: Does this antibody exist in the protein data?
        if isinstance(antibody, str):
            try:
                # Drop column from protein data
                self._protein_matrix.drop(antibody, axis=1, inplace=True)
                print(f"Removed {antibody} from protein data.")
            except:
                raise AntibodyLookupError(f"'{antibody}' not found in protein data.")
        else:
            raise AntibodyLookupError("Antibody must be a string")

        # CHECK: Does this antibody have a fit?
        if self._all_fits_dict != None and antibody in self._all_fits_dict:
            self._all_fits_dict.pop(antibody)
            print(f"Removed {antibody} fits.")
    
    def fit_antibody(self,
                     input: list | str,
                     ab_name: str = None,
                     transform_type: str = None,
                     transform_scale: int = 1,
                     model: str = 'gaussian',
                     plot: bool = False,
                     **kwargs) -> dict:
        """Fits a mixture model to an antibody and returns its optimal parameters.

        This function can be called to either initially fit a single antibody
        with a mixture model or replace an existing fit. This function can be called
        after fit_all_antibodies has been called to replace individual fits.

        Args:
            input (list | str): raw values from protein data or antibody name 
            ab_name (str, optional): name of antibody. Ignore if calling 
                fit_antibody by supplying the antibody name in the "input" parameter.
            transform_type (str): type of transformation. "log" or "arcsinh"
            transform_scale (int): multiplier applied during transformation
            model (str): type of model to fit. "gaussian" or "nb"
            plot (bool): option to plot each model
            **kwargs: initial arguments for sklearn's GaussianMixture (optional)

        Raises:
            InvalidModelError: The provided model is neither "gaussian" nor "nb".
            ExtraArgumentsError: Additional kwargs can only be provided for "gaussian".
            TransformTypeError: The transform type is neither "log" nor "arcsinh".
            PlotAntibodyFitError: Plot must be a boolean value.
            AntibodyLookupError: The provided antibody name is not found in the protein data.
            TransformTypeError: A transform scale cannot be provided without a transform type.

        Returns:
            dict: results from optimization as either gauss_params/nb_params.
        """

        # Checking parameters
        if model != 'nb' and model != 'gaussian':
            raise InvalidModelError(("Invalid model. Please choose 'gaussian' or 'nb'. "
                                    "Default: 'gaussian'."))

        if model == 'nb' and len(kwargs) > 0:
            raise ExtraArgumentsError("additional kwargs can only be used for 'gaussian'.")

        if transform_scale != 1 and transform_type is None:
            raise TransformTypeError("transform_type must be chosen to use "
                                  "transform_scale. choose 'log' or 'arcsinh'.")

        # if isinstance(transform_scale, int) == False:
        #     raise TransformScaleError("'transform_scale' must be an integer value.")

        if isinstance(plot, bool) == False:
            raise PlotAntibodyFitError("'plot' must be a boolean value.")

        # Check if all_fits_dict exists
        if self._all_fits_dict is None:
            self._all_fits_dict = {}
            for ab in list(self._protein_matrix.columns):
                self._all_fits_dict[ab] = None

        # Indicate whether this function call is for individual fitting
        individual = False

        # Fitting a single antibody using its name
        if isinstance(input, str):
            # if input in self._protein_matrix.columns:
            try:
                data_vector = list(self._protein_matrix.loc[:, input].values)
                # Also set ab_name to input, since input is the string of the antibody
                ab_name = input
                individual = True

                if transform_type is None:
                    # If no transform type, reset data back to normal
                    self.protein.loc[:, ab_name] = self._temp_protein.loc[:, ab_name]
                    data_vector = list(self._temp_protein.loc[:, ab_name])

            except:
                raise AntibodyLookupError(f"'{input}' not found in protein data.")
        # Fitting all antibodies at once
        else:
            data_vector = input

        if transform_type is not None:
            if transform_type == 'log':
                data_vector = _log_transform(d_vect=data_vector,
                                            scale=transform_scale)
                self.protein.loc[:, ab_name] = data_vector

            elif transform_type == 'arcsinh':
                data_vector = _arcsinh_transform(d_vect=data_vector,
                                                scale=transform_scale)
                self.protein.loc[:, ab_name] = data_vector

            else:
                raise TransformTypeError(("Invalid transformation type. " 
                                          "Please choose 'log' or 'arcsinh'. "
                                          "Default: None."))

        if model == 'gaussian':
            gauss_params = _gmm_results(counts=data_vector,
                                        ab_name=ab_name,
                                        plot=plot,
                                        **kwargs)

            # Check if a params already exists in dict
            if individual:
                # Add or replace the existing fit so far
                self._all_fits_dict[input] = gauss_params

                # Update all of the fits to self._all_fits
                # while filtering out None (for antibodies without fits yet)
                self._all_fits = list(filter(None, self._all_fits_dict.values()))

            return gauss_params

        elif model == 'nb':
            nb_params = _nb_mle_results(counts=data_vector,
                                        ab_name=ab_name,
                                        plot=plot)

            if individual:
                self._all_fits_dict[input] = nb_params
                self._all_fits = list(filter(None, self._all_fits_dict.values()))

            return nb_params
            