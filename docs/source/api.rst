API
===
Does this create a new page?

Testing to see if it references IPD object on another page

To retrieve a list of random ingredients,
you can use the ``immunopheno.plots.plot_UMAP()`` function:

.. autofunction:: immunopheno.plots.plot_UMAP

The ``kind`` parameter should be either ``"meat"``, ``"fish"``,
or ``"veggies"``. Otherwise, :py:func:`immunopheno.plots.plot_UMAP`
will raise an exception.


For example:

>>> import immunopheno
>>> immunopheno.plots.plot_UMAP()
['shells', 'gorgonzola', 'parsley']

Testing new google docstring for module level functions

.. autofunction:: immunopheno.plots.plot_UMAP

.. autofunction:: immunopheno.plots.plot_antibody_correlation