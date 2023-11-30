.. obstacle map documentation master file, created by
   sphinx-quickstart on Thu Nov 30 12:36:59 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Документация по построению карты препятствий
=============================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:



.. Indices and tables
.. ==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Сокеты
======================

.. toctree::
   :maxdepth: 4
   :caption: Нужны для получения и отправления результатов в "микросервисы"

   sockets.rst


Построение карты препятствий
==============================

.. toctree::
   :maxdepth: 4
   :caption: Делаем сегментацию снимка при помощи SAM и строим карту препятствий

   sam_and_model.rst

Детекция
=========

.. toctree::
   :maxdepth: 4
   :caption: Детекция объектов по классам живое/не живое

   yolo.rst