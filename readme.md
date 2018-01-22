# Introduction to CUDA (2)

The aim of this project is to understand the basics of CUDA.
While this project focus on the use of patterns, [this one](https://github.com/EspeuteClement/INSA-5IF-CUDA) focus on the very basis.

The project is available on [the associated GitHub repository](https://github.com/Sn0wFox/INSA-5IF-CUDA2),
and you can get it with the following command:

```
git clone https://github.com/Sn0wFox/INSA-5IF-CUDA2
```

## Prerequisites

At the moment, this project **only works on windows using Visual Studio (2017) and Cuda (9.0)**.
We're not planning to support anything else.

## Run the project

Simply clone it and open the .sln with Visual Studio. You will find two projects:
1. In the folder `pattern1`, you will find the code related to the first exercise, i.e. a simple histogramme of characters in a list.
2. In the folder `pattern2`, you will find the code related to the second exercise, i.e. the creation of a stencil.

NOTE: unfortunately, we had trouble with the code of the second and the third exercise, which was not working out of the box.
We spent a lot of time on the second exercise, the third being easily found on the internet (and newer versions of Cuda have a way to make a reduction out of the box),
but sadly it still doesn't work properly.

## Contributors

- ESPEUTE Clément
- PERICAS-MOYA Ruben