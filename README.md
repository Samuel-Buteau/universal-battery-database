# Universal Battery Database

The Universal Battery Database is an open source software for managing Lithium-ion cell data. Its primary purposes are:
1. Organize and parse experimental measurement (e.g. long term cycling and electrochemical impedance spectroscopy) data files of Lithium-ion cells.
2. Perform sophisticated modelling using machine learning and physics-based approaches.
3. Describe and organize the design and chemistry information of cells (e.g. electrodes, electrolytes, geometry), as well as experimental conditions (e.g. temperature).
4. Automatically refresh a database as new data comes in.
5. Visualize experimental results.
6. Quickly search and find data of interest.
7. Quality control.

The Universal Battery Database was developed at the [Jeff Dahn Research Group](https://www.dal.ca/diff/dahn/about.html) at Dalhousie University.

## Table of Contents

- [Installation](#installation)
  * [Prerequisites](#prerequisites)
  * [Installing Dependencies and Configuring Environment](#installing-dependencies-and-configuring-environment)
- [Using the Software](#using-the-software)
  * [ML Smoothing](#ml-smoothing)
- [Theoretical Physics and Computer Science Behind the Software](#theoretical-physics-and-computer-science-behind-the-software)

## Installation

### Prerequisites

- [Python 3](https://www.python.org/downloads/)
- [pip and virtualenv](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/)


### Installing Dependencies and Configuring Environment

#### 1. Create and Activate a new virtual environment.

cmd (Windows):
```cmd
>py -m venv env
>.\env\Scripts\activate
```

Bash (macOS and Linux):
```bash
$ python3 -m venv env
$ source env/bin/activate
```


#### 2. Install requirements

If you do not have a database, install requirements with:
```bash
pip3 install -r requirements_nosql.txt
```

Otherwise, install requirements with:
```bash
pip3 install -r requirements.txt
```


#### 3. [Install PostgreSQL](https://www.2ndquadrant.com/en/blog/pginstaller-install-postgresql/).

**Make sure the installation includes the PostgreSQL Unicode ODBC driver** (e.g. ODBC 64-bitODBC 64-bit).

Follow the installation instructions and create new user and password.

#### 4. Add the bin path of the install to the Path variable.

#### 5. Run

```bash
psql -U postgres
```

```sql
CREATE DATABASE my_project;

CREATE USER my_user WITH PASSWORD ‘my_password’;

GRANT ALL PRIVILEGES ON DATABASE my_project TO my_user;
```


#### 6. Create `config.ini` in the root directory.

`config.ini` should contain the following (feel free to modify the values):

```
[DEFAULT]
Database = database
User = user
Password = password
Host = localhost
Port = 5432
```

This is for security purposes.

#### 7. Download a dataset file and put it in the appropriate folder.

#### 8. Create `neware_parser/config.ini`.

`neware_parser/config.ini` should contain the following (again, feel free to modify the values):

```
[DEFAULT]
Database = database
User = user
Password = password
Host = localhost
Port = 0000
Backend = sqlite3
SecretKey = your_very_secret_key
```


## Using the Software

To quickly see the web page and start developing, run
```bash
python3 manage.py runserver 0.0.0.0:8000
```
then visit `http://localhost:8000/` with a web browser.

When running the code in production, run
```bash
python3 manage.py process_tasks
```
in a separate terminal to allow background tasks (such as parsing data files). 
This will process the tasks as they are defined.

### ML Smoothing
cmd (Windows)
```cmd
>ml_smoothing.bat
````

Bash (macOS and Linux)
```Bash
$ sh ml_smoothing.sh path-figures dataset-version "optional-note-to-self"
```

## Theoretical Physics and Computer Science Behind the Software

We hypothesize that we can make [good generalizations](https://github.com/Samuel-Buteau/universal-battery-database/wiki/Generalization-Criteria) by [approximating](https://github.com/Samuel-Buteau/universal-battery-database/wiki/The-Universal-Approximation-Theorem) the functions that map one degradation mechanism to another using neural networks. 

We aim to develop a theory of lithium-ion cells. We first break down the machine learning problem into smaller sub-problems. From there, we development frameworks to convert the theory to practical implementations. Finally, we apply the method to experimental data and evaluate the result.