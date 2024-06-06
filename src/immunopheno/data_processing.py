import pandas as pd
import anndata

class ImmunoPhenoError(Exception):
    """A base class for ImmunoPheno Exceptions."""

class PlotUMAPError(ImmunoPhenoError):
    """No normalized counts are found when plotting a normalized UMAP"""

class ImmunoPhenoData:
    """A class to hold single-cell data (CITE-Seq, etc) and cytometry data.
    
    Performs fitting of gaussian/negative binomial mixture models and
    normalization to antibodies present in a protein dataset. Requires protein
    data to be supplied using the protein_matrix or scanpy field.

    Args:
        protein_matrix (str | pd.Dataframe): file path or dataframe to ADT count/protein matrix. 
            Format: Row (cells) x column (antibodies/proteins).
        gene_matrix (str | pd.DataFrame): file path or dataframe to UMI count matrix.
            Format: Row (cells) x column (genes).
        cell_labels (str | pd.DataFrame): file path or dataframe to cell type labels. 
            Format: Row (cells) x column (cell type such as Cell Ontology ID). Must contain 
            a column called "labels".
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
            ImmunoPhenoData: ImmunoPhenoData object with modified dataframes 
            based on provided rows/cells names.
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
            pd.DataFrame: Dataframe containing protein data.
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
            pd.DataFrame: Dataframe containing RNA data.
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
            dict: Key-value pairs represent an antibody name with a
            nested dictionary containing the respective mixture model fits. 
            Fits are ranked starting with the lowest AIC.
        """
        return self._all_fits_dict

    @property
    def normalized_counts(self) -> pd.DataFrame:
        """Get the normalized protein dataframe in the object.

        This dataframe will only be present if normalize_all_antibodies 
        has been called. The format will have rows (cells) and columns (proteins/antibodies).
        Note that some rows may be missing/filtered out from the normalization step.

        Returns:
            pd.DataFrame: Normalized protein counts for each antibody. 
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
            pd.DataFrame: Dataframe containing two columns: "labels" and "celltypes". 
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

    def convert_labels(self) -> None:
        """Convert all cell ontology IDs to a common name.

        Requires all values in the "labels" column of the cell labels dataframe to
        follow the cell ontology format of CL:0000000 or CL_0000000.

        Returns:
            None. Modifies the cell labels dataframe in-place.
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
        """Removes an antibody from all protein data and mixture model fits.

        Removes all values for an antibody from all protein dataframes in-place. If
        fit_antibody or fit_all_antibodies has been called, it will also remove 
        the mixture model fits for that antibody.

        Args:
            antibody (str): name of antibody to be removed.

        Returns:
            None. Modifies all protein dataframes and fits data in-place.
        """
        if not isinstance(antibody, str):
            raise AntibodyLookupError("Antibody must be a string")
        
        column_found = False

        # Iterate through all attributes of the class
        for attr_name, attr_value in self.__dict__.items():
            # Check if the attribute is a DataFrame
            if isinstance(attr_value, pd.DataFrame):
                # Try to drop the column if it exists
                if antibody in attr_value.columns:
                    attr_value_copy = attr_value.copy()  # Create a copy to avoid SettingWithCopyWarning
                    attr_value_copy.drop(antibody, axis=1, inplace=True)
                    setattr(self, attr_name, attr_value_copy)
                    column_found = True

        if not column_found:
            raise AntibodyLookupError(f"'{antibody}' not found in protein data.")
        else:
            print(f"Removed {antibody} from object.")

        # Reset the regular and normalized UMAPs after removing an antibody from the protein dataset
        self._raw_umap = None
        self._norm_umap = None

        # CHECK: Does this antibody have a fit?
        if self._all_fits_dict != None and antibody in self._all_fits_dict:
            self._all_fits_dict.pop(antibody)
            print(f"Removed {antibody} fits.")
    
    def select_mixture_model(self,
                             antibody: str,
                             mixture: int) -> None:
        """Overrides the best mixture model fit for an antibody.

        Args:
            antibody (str): name of antibody to modify best mixture model fit.
            mixture (int): preferred number of mixture components to override a fit.

        Returns:
            None. Modifies mixture model order in-place.
        """
        # CHECK: is mixture between 1 and 3
        if (not 1 <= mixture <= 3):
            raise BoundsThresholdError("Number for Mixture Model must lie between 1 and 3 (inclusive).")

        # CHECK: Does this antibody have a fit?
        if self._all_fits_dict != None and antibody in self._all_fits_dict:

            # Find current ordering of mixture models for this antibody
            # We know the element at index 0 is by default the "best" (sorted by lowest AIC)
            mix_order_list = list(self._all_fits_dict[antibody].keys())

            # Find the index of the element we CHOOSE to be the best
            choice_index = mix_order_list.index(mixture)

            # SWAP the ordering of these two elements in the list
            mix_order_list[0], mix_order_list[choice_index] = mix_order_list[choice_index], mix_order_list[0]

            # With this new list ordering, re-create the dictionary
            reordered_dict = {k: self._all_fits_dict[antibody][k] for k in mix_order_list}

            # Re-assign this dictionary to this antibody key
            self._all_fits_dict[antibody] = reordered_dict
        else:
            # Else, we cannot find the antibody's fits
            raise AntibodyLookupError(f"{antibody} fits cannot be found.")
    
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
        after fit_all_antibodies has been called to modify individual fits.

        Args:
            input (list | str): raw values from protein data or antibody name.
            ab_name (str, optional): name of antibody. Ignore if calling
                fit_antibody by supplying the antibody name in the "input" parameter.
            transform_type (str): type of transformation. "log" or "arcsinh".
            transform_scale (int): multiplier applied during transformation.
            model (str): type of model to fit. "gaussian" or "nb".
            plot (bool): option to plot each model.
            **kwargs: initial arguments for sklearn's GaussianMixture (optional).

        Returns:
            dict: Results from optimization as either gauss_params/nb_params.
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
    
    def fit_all_antibodies(self,
                           transform_type: str = None,
                           transform_scale: int = 1,
                           model: str = 'gaussian',
                           plot: bool = False,
                           **kwargs) -> None:
        """Fits all antibodies with a Gaussian or Negative Binomial mixture model.

        Fits a Gaussian or Negative Binomial mixture model to all antibodies
        in the protein dataset. After all antibodies are fit, the output will 
        display the number of each mixture model fit in the dataset. This includes
        the names of the antibodies that were fit with a single component model.
        
        Args:
            transform_type (str): type of transformation. "log" or "arcsinh".
            transform_scale (int): multiplier applied during transformation.
            model (str): type of model to fit. "gaussian" or "nb".
            plot (bool): option to plot each model.
            **kwargs: initial arguments for sklearn's GaussianMixture (optional).
        
        Returns:
            None. Results will be stored in the class. This is accessible using 
            the "fits" property.
        """

        fit_all_results = []

        for ab in tqdm(self._protein_matrix, total=len(self._protein_matrix.columns)):
            # if plot: # Print antibody name if plotting
                # print("Antibody:", ab)
            fits = self.fit_antibody(input=self._protein_matrix.loc[:, ab],
                                    ab_name=ab,
                                    transform_type=transform_type,
                                    transform_scale=transform_scale,
                                    model=model,
                                    plot=plot,
                                    **kwargs)
            self._all_fits_dict[ab] = fits
            fit_all_results.append(fits)
        
        number_3_component = 0
        number_2_component = 0
        all_1_component = []

        for ab_name, fit_results in self._all_fits_dict.items():
            num_components = next(iter(fit_results))
            if num_components == 3:
                number_3_component += 1
            elif num_components == 2:
                number_2_component += 1
            elif num_components == 1:
                all_1_component.append(ab_name)
                
        print("Number of 3 component models:", number_3_component)
        print("Number of 2 component models:", number_2_component)
        print("Number of 1 component models:", len(all_1_component))

        if len(all_1_component) > 0:
            print("Antibodies of 1 component models:")
            for background_ab in all_1_component:
                print(background_ab)

        # Store in class
        self._all_fits = fit_all_results

    def normalize_all_antibodies(self,
                                 p_threshold: float = 0.05,
                                 sig_expr_threshold: float = 0.85,
                                 bg_expr_threshold: float = 0.15,
                                 bg_cell_z_score: int = 10,
                                 cumulative: bool = False) -> None:
        
        """Normalizes all antibodies in the protein data.

        The normalization step uses the fits from the mixture model to remove 
        background noise from the overall signal expression of an antibody. This will take into
        account non-specific antibody binding if RNA data is present. If RNA data 
        is present, the effects of cell size on the background noise will be regressed out
        for cells not expressing the antibody. Likewise, if cell labels are provided, 
        the effects of cell types on the background noise for these cells will also be regressed out. 
        These effects are determined by performing a linear regression using the 
        total number of mRNA UMI counts as a proxy for cell size.

        Args:
            p_threshold (float): level of significance for testing the association
                between cell size/type and background noise from linear regression. If 
                the p-value is smaller than the threshold, these factors are regressed out.
            sig_expr_threshold (float): cells with a percentage of expressed proteins above
                this threshold are filtered out.
            bg_expr_threshold (float): cells with a percentage of expressed proteins below
                this threshold are filtered out.
            bg_cell_z_score (int): The number of standard deviations of average protein expression
                to separate cells that express an antibody from cells that do not express an antibody.
                A larger value will result in more discrete clusters in the normalized 
                protein expression space.
            cumulative (bool): flag to indicate whether to return the 
                cumulative distribution probabilities.
        
        Returns:
            None. Results will be stored in the class. This is accessible using 
            the "normalized_counts" property.
        """

        # Check if parameters have changed
        if (p_threshold, sig_expr_threshold, 
            bg_expr_threshold, bg_cell_z_score, cumulative) != self._last_normalize_params:
            # If so, reset UMAP stored in class
            self._norm_umap = None
            # Update the parameters
            self._last_normalized_params = (p_threshold, sig_expr_threshold, 
                                            bg_expr_threshold, bg_cell_z_score, cumulative)
        
        bg_cell_z_score = -bg_cell_z_score
        if self._all_fits is None:
            raise EmptyAntibodyFitsError("No fits found for each antibody. Please "
                                         "call fit_all_antibodies() or fit_antibody() first.")

        if None in self._all_fits_dict.values():
            raise IncompleteAntibodyFitsError("All antibodies must be fit before normalizing. "
                                              "call fit_all_antibodies() or fit_antibody() for "
                                              "each antibody.")

        all_fits = self._all_fits_dict

        if (not 0 <= p_threshold <= 1 or
            not 0 <= sig_expr_threshold <= 1 or
            not 0 <= bg_expr_threshold <= 1):
            raise BoundsThresholdError("threshold must lie between 0 and 1 (inclusive)")

        # if not bg_cell_z_score < 0:
        #     raise BackgroundZScoreError("bg_cell_z_score must be less than 0")

        warnings.filterwarnings('ignore')
        # Classify all cells as either background or signal
        classified_cells = _classify_cells_df(all_fits, self._protein_matrix)

        # Filter out cells that have a high signal: background ratio (default: 1.0)
        classified_cells_filt = _filter_classified_df(classified_cells,
                                                    sig_threshold=sig_expr_threshold,
                                                    bg_threshold=bg_expr_threshold)
        self._classified_filt_df = classified_cells_filt

        # Filter the same cells from the protein data
        protein_cleaned_filt = _filter_count_df(classified_cells_filt,
                                                self._protein_matrix)

        # Filter from cell labels if dealing with single cell data
        if self._cell_labels is not None:
            cell_labels_filt = _filter_cell_labels(classified_cells_filt,
                                                        self._cell_labels)
            self._cell_labels_filt_df = cell_labels_filt # this will replace the norm label field directly

        # Calculate z scores for all values
        z_scores = _z_scores_df(all_fits, protein_cleaned_filt)
        self._z_scores_df = z_scores

        # Extract z scores for background cells
        background_z_scores = _bg_z_scores_df(classified_cells_filt, z_scores)

        # Set cumulative flag
        self._cumulative = cumulative

        # Set the background cell z score to the class attribute (for STvEA)
        self._background_cell_z_score = bg_cell_z_score

        # The server currently uses +10 adjustment to all background cells
        # Calculate the additional adjustment from the user-provided background cell z score
        # Example: bg_cell_z_score: -10, stvea_correction_value: 0
        # Example: bg_cell_z_score: -3, stvea_correction_value: 7
        self._stvea_correction_value = 10 + bg_cell_z_score

        # If dealing with single cell data with cell_labels,
        # Run linear regression to regress out z scores based on size and cell type
        if self._gene_matrix is not None and self._cell_labels is not None:
            df_by_type = _z_avg_umi_sum_by_type(background_z_scores,
                                                self._gene_matrix,
                                                cell_labels_filt)

            lin_reg_type = _linear_reg_by_type(df_by_type)
            self._linear_reg_df = lin_reg_type

            # Normalize all protein values
            normalized_df = _normalize_antibodies_df(
                                    protein_cleaned_filt_df=protein_cleaned_filt,
                                    fit_all_results=all_fits,
                                    p_threshold=p_threshold,
                                    background_cell_z_score=bg_cell_z_score,
                                    classified_filt_df=classified_cells_filt,
                                    cell_labels_filt_df=cell_labels_filt,
                                    lin_reg_dict=lin_reg_type,
                                    cumulative=cumulative)

        # If dealing with single cell data WITHOUT cell labels:
        # Run linear regression to regress out only size
        elif self._gene_matrix is not None and self._cell_labels is None:
            z_umi = _z_avg_umi_sum(background_z_scores, self._gene_matrix)
            lin_reg = _linear_reg(z_umi)
            self._linear_reg_df = lin_reg

            # Normalize all protein values
            normalized_df = _normalize_antibodies_df(
                                    protein_cleaned_filt_df=protein_cleaned_filt,
                                    fit_all_results=all_fits,
                                    p_threshold=p_threshold,
                                    background_cell_z_score=bg_cell_z_score,
                                    classified_filt_df=classified_cells_filt,
                                    lin_reg=lin_reg,
                                    cumulative=cumulative)

        # Else, normalize values for flow cytometry data
        else:
            # Normalize all values in the protein matrix
            normalized_df = _normalize_antibodies_df(
                                    protein_cleaned_filt_df=protein_cleaned_filt,
                                    fit_all_results=all_fits,
                                    p_threshold=p_threshold,
                                    background_cell_z_score=bg_cell_z_score,
                                    classified_filt_df=classified_cells_filt,
                                    cumulative=cumulative)
        
        self._normalized_counts_df = normalized_df