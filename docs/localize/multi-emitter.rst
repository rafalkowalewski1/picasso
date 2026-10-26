.. _localize-multi-emitter:

Overlapping Emitters (Multi-Emitter Fitting)
============================================

Every other fit of Picasso Localize places a box around each identified spot and fits **one** emitter to it. Two emitters closer than about two PSF widths in the same frame then give either one localization between them or a poor fit (a high ``reduced_chi_square``). This happens more frequently in data with high density of blinking.

The **multi-emitter fit** localizes such emitters separately. Picasso implements the 3D-DAOSTORM analysis of `Babcock, et al. Optical Nanoscopy, 2012 <https://doi.org/10.1186/2192-2853-1-6>`_ in its fixed-width form, ported from its reference implementation, `storm-analysis <https://github.com/ZhuangLab/storm-analysis>`_, whose license is reproduced in `LICENSES/storm-analysis-LICENSE.txt <https://github.com/jungmannlab/picasso/blob/master/LICENSES/storm-analysis-LICENSE.txt>`__. Picasso adds three things: the PSF is integrated over the pixels, as Simulate's movies are, instead of sampled at the pixel centers; the PSF width can be estimated from the movie; and every emitter gets a localization precision.

It fits whole frames instead of boxes, with all their emitters at once:

1. The **background** of the frame is estimated by a broad Gaussian low pass of the frame minus the current fit.
2. **Peaks** are found in what the fit does not explain yet: the frame minus the background and the fit, smoothed with the PSF and divided by its expected noise, i.e., in units of the signal-to-noise ratio (SNR). A peak must exceed the ``SNR threshold`` and be the maximum within the ``Peak search radius``.
3. The new peaks join the fit. Each emitter is fitted by maximum likelihood, with every other emitter in place; emitters near a new or a removed one are fitted again until all have converged.
4. Of two emitters closer than the PSF width, the dimmer one is removed, and so is every emitter whose significance does not exceed the ``SNR threshold``.

The cycles repeat until a cycle finds no new peak, or up to ``Max. cycles``. The PSF is a 2D Gaussian integrated over the pixels, with a width (``sx = sy``) fixed for the whole movie.


.. _localize-multi-emitter-when:

When to use it
--------------

As per the 2D datasets of the SMLM Challenge 2016 (`Sage et al., Nature Methods, 2019 <https://doi.org/10.1038/s41592-019-0364-4>`_), use it for **dense 2D data with the emitters in focus**. It also should do no harm on sparse frames when the parameters are set right.

Consider the limits:

- **Close pairs.** Of two emitters closer than the PSF width only the brighter is kept, so pairs closer than about 1.5 PSF widths are seldom resolved; pairs two widths apart mostly are.
- **Emitters out of focus.** The width is fixed, so a spot wider than the PSF is fitted as several narrow emitters. On the SMLM Challenge's 1.5 µm thick sample, emitters more than 200 nm out of focus got about one false neighbor each. Do not use it for 3D or thick samples.
- **Bright emitters and non-Gaussian PSFs.** The rings of a real PSF around a very bright emitter can be found as faint emitters next to it. Raise the ``SNR threshold``, or filter by ``significance`` afterwards (see :ref:`filter-expression`).
- **Structured background.** Out-of-focus fluorescence that changes from frame to frame is found as dim emitters wherever the background filter does not follow it. Tune the threshold with the preview, or filter by ``significance`` afterwards.
- It runs on the CPU, frame by frame on several threads, and fits one channel at a time (no multichannel or split-FOV fitting yet).

.. _localize-multi-emitter-gui:

In Picasso: Localize
--------------------

In the ``Parameters`` dialog, under ``Fit Settings``, select the model ``2D Gaussian, overlapping emitters``. The optimizer is MLE; the identification settings (box, method, filters) disappear, since the fit finds the spots itself, while the ROIs and the frame range still apply. Its settings are under ``Overlapping emitters``:

- ``SNR threshold`` (default 6): the peak threshold and the minimum significance of an emitter.
- ``PSF sigma (nm)``: with ``Estimate`` ticked (the default), the width is estimated from the movie as the median width of isolated emitters fitted with a free width; the estimate is shown once the fit is done. Untick ``Estimate`` to give the width.
- ``Max. cycles`` (20), ``Peak search radius`` (5 camera pixels) and ``Background filter sigma`` (8 camera pixels): rarely changed.

Run ``Analyze`` > ``Localize (Identify & Fit)`` or ``Analyze`` > ``Fit``; there is nothing to identify first. With ``Preview`` ticked, the current frame is fitted with the settings shown, so the threshold can be tuned frame by frame. Emitters with a neighbor within four PSF widths are marked with a sky blue ``+``, the others with the usual green ``x``.

.. _localize-multi-emitter-output:

Output
------

One localization per emitter, with the columns of an MLE fit (see :ref:`files-localization-hdf5`) and two more:

- ``significance``: the signal of the emitter over the square root of the background it sits on, both weighted by the PSF.
- ``n_emitters``: the number of emitters within four PSF widths, the emitter itself included; 1 for an isolated emitter. To keep the isolated ones only, filter by ``n_emitters == 1``.

``sx`` and ``sy`` hold the fixed PSF width. The localization precisions ``lpx`` and ``lpy`` and ``photons_unc`` and ``bg_unc`` are the Cramer-Rao bounds from the Fisher matrix of the emitter and its neighbors within four PSF widths, so they grow where emitters overlap. Where that matrix is singular, the correction of `Huang et al., Biomed. Opt. Express, 2011 <https://doi.org/10.1364/BOE.2.001377>`_ (Eq. 11) is applied. ``log_likelihood`` and ``reduced_chi_square`` describe the fit of the region around the emitter, its neighbors included. The settings, and the PSF width that was used, are recorded in the ``Fit multi-emitter`` entry of the metadata.

.. _localize-multi-emitter-cli:

Command line
------------

.. code-block:: console

   picasso localize movie.tif --multi-emitter -px 130 -bl 100 -s 0.5

Further options: ``--snr-threshold`` (6), ``--psf-sigma`` (nm; estimated by default), ``--max-cycles`` (20), ``--peak-radius`` (5), ``--background-sigma`` (8) and ``--seed-identification``, which starts each frame from the spots the identification options find instead of the first cycle's peaks. See :ref:`cmd-localize`.

.. dropdown:: Python
   :icon: code
   :class-container: api-example

   ``localize_multi_emitter`` takes the movie, its metadata and the camera
   information, and the settings as one
   ``localize.MultiEmitterParameters``; without them, the defaults apply
   and the PSF width is estimated.

   .. code-block:: python

      from picasso import io, localize

      movie, info = io.load_movie("movie.tif")
      camera_info = {
          "Baseline": 100, "Sensitivity": 0.5, "Gain": 1, "Pixelsize": 130
      }
      locs, info = localize.localize_multi_emitter(
          movie,
          info,
          camera_info=camera_info,
          parameters=localize.MultiEmitterParameters(snr_threshold=6.0),
          progress="console",
      )
      io.save_locs("movie_locs.hdf5", locs, info)
      # the PSF width the fit used, in nm
      print(info[-1]["PSF Sigma (nm)"])

References
----------

- Babcock, H., Sigal, Y. M. & Zhuang, X. A high-density 3D localization algorithm for stochastic optical reconstruction microscopy. *Optical Nanoscopy* 1, 6 (2012). https://doi.org/10.1186/2192-2853-1-6
- storm-analysis, Zhuang Lab, Harvard University, v2.4.1. https://github.com/ZhuangLab/storm-analysis
- Huang, F., Schwartz, S. L., Byars, J. M. & Lidke, K. A. Simultaneous multiple-emitter fitting for single molecule super-resolution imaging. *Biomed. Opt. Express* 2, 1377–1393 (2011). https://doi.org/10.1364/BOE.2.001377
- Sage, D. et al. Super-resolution fight club: assessment of 2D and 3D single-molecule localization microscopy software. *Nature Methods* 16, 387–395 (2019). https://doi.org/10.1038/s41592-019-0364-4
