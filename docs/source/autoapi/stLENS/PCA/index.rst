stLENS.PCA
==========

.. py:module:: stLENS.PCA


Classes
-------

.. autoapisummary::

   stLENS.PCA.PCA


Module Contents
---------------

.. py:class:: PCA(device=None, data=None)

   .. py:attribute:: device
      :value: None



   .. py:attribute:: data
      :value: None



   .. py:method:: fit(X=None, eigen_solver='wishart')


   .. py:method:: get_signal_components(n_components=0)


   .. py:method:: _wishart_matrix(X)


   .. py:method:: to_gpu(Y)


   .. py:method:: _get_eigen(X)


   .. py:method:: _random_matrix(X)


   .. py:method:: plot_mp(comparison=False, path=False, info=True, bins=None, title=None)


