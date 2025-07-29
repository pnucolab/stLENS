stLENS.calc
===========

.. py:module:: stLENS.calc


Classes
-------

.. autoapisummary::

   stLENS.calc.Calc


Module Contents
---------------

.. py:class:: Calc(device=None, data=None, L=None, L_mp=None)

   .. py:attribute:: L
      :value: []



   .. py:attribute:: V
      :value: None



   .. py:attribute:: L_mp
      :value: None



   .. py:attribute:: explained_variance_
      :value: []



   .. py:attribute:: total_variance_
      :value: []



   .. py:attribute:: device
      :value: None



   .. py:attribute:: data
      :value: None



   .. py:method:: style_mp_stat()


   .. py:method:: _tw(rmt_device)

      Tracy-Widom critical eigenvalue



   .. py:method:: _mp_parameters(L, rmt_device)


   .. py:method:: _marchenko_pastur(x, dic)

      Distribution of eigenvalues



   .. py:method:: _mp_pdf(x, L, rmt_device)

      Marchnko-Pastur PDF



   .. py:method:: _mp_calculation(L, Lr, rmt_device, eta=1, eps=10**(-6), max_iter=1000)


   .. py:method:: _cdf_marchenko(x, dic)


   .. py:method:: _call_mp_cdf(L, dic)

      CDF of Marchenko Pastur



