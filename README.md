# TIMEX
[![Tests with PyTest](https://github.com/AlexMV12/TIMEX/actions/workflows/run_tests.yml/badge.svg)](https://github.com/AlexMV12/TIMEX/actions/workflows/run_tests.yml)
![Coverage](badges/coverage.svg)
![PyPI](https://img.shields.io/pypi/v/timexseries)
![PyPI - Downloads](https://img.shields.io/pypi/dm/timexseries)

TIMEX (referred in code as `timexseries`) is a framework for time-series-forecasting-as-a-service.

Its main goal is to provide a simple and generic tool to build websites and, more in general,
platforms, able to provide the forecasting of time-series in the "as-a-service" manner.

This means that users should interact with the service as less as possible.

An example of the capabilities of TIMEX can be found at [covid-timex.it](https://covid-timex.it)  
That website is built using the [Dash](https://dash.plotly.com/), on which the visualization
part of TIMEX is built. A deep explanation is available in the 
[dedicated repository](https://github.com/AlexMV12/covid-timex.it).

## Installation
The main two dependencies of TIMEX are [Facebook Prophet](https://github.com/facebook/prophet)
and [PyTorch](https://pytorch.org/). 
If you prefer, you can install them beforehand, maybe because you want to choose the CUDA/CPU
version of Torch.

However, installation is as simple as running:

`pip install timexseries`

## Get started
Please, refer to the Examples folder. You will find some Jupyter Notebook which illustrate
the main characteristics of TIMEX. A Notebook explaining the covid-timex.it website is present,
along with the source code of the site, [here](https://github.com/AlexMV12/covid-timex.it).

## Documentation
The full documentation is available at [here](https://alexmv12.github.io/TIMEX/timexseries/index.html).

## Contacts
If you have questions, suggestions or problems, feel free to open an Issue.
You can contact us at:

- alessandro.falcetta@polimi.it
- manuel.roveri@polimi.it

