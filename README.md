cookiecutter project for simple RTS data analysis.


# Cookiecutter python projects

_A logical, reasonably standardized, but flexible project structure for doing and sharing data analysis._


#### [Project homepage](https://github.com/alfgarciagarcia/cookiecutter_rts_analysis)


### Requirements to use the cookiecutter template:
-----------
 - Python 3.9 higher
 - [Cookiecutter Python package](http://cookiecutter.readthedocs.org/en/latest/installation.html) >= 1.4.0: This can be installed with pip by or conda depending on how you manage your Python packages:

``` bash
$ pip install cookiecutter
```

or

``` bash
$ conda config --add channels conda-forge
$ conda install cookiecutter
```


### To start a new project, run from the :
------------

    cookiecutter cookiecutter_rts_analysis
    cookiecutter https://github.com/alfgarciagarcia/cookiecutter_rts_analysis




### The resulting directory structure
------------

The directory structure of your new project looks like this: 

```
├── LICENSE
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   ├── raw            <- The original, immutable data dump.
│   ├── test	          <- Data to test models
│   ├── train	         <- Data to train models
|   └── reports        <- Generated analysis as HTML, PDF, Excel, etc.
│       └── figures    <- Generated graphics and figures to be used in reporting
├── docs               <- A default Sphinx project; see sphinx-doc.org for details
│
├── models             <- Trained and serialized models, model predictions, or model 	  │			   summaries
│


```

