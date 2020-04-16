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
    + [Without Database Install](#without-database-install)
    + [With Database Install](#with-database-install)
- [Using the Software](#using-the-software)
  * [ML Smoothing](#ml-smoothing)
- [Theoretical Physics and Computer Science Behind the Software](#theoretical-physics-and-computer-science-behind-the-software)

## Installation

### Prerequisites

- [Python 3](https://www.python.org/downloads/)
- [pip and virtualenv](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/)


### Installing Dependencies and Configuring Environment
There are two distinct possibilities: 
1. if you want to use the database features such as parsing and organising experimental data and metadata, __you need to install with a database__.
2. if you only want to play around with modelling, using a dataset compiled somewhere else, __you do not need to install with a database__. Note that you can always install with a database if unsure, but it is more involved.

Either way, you need to create and activate a new virtual environment to start (note: `env` can be any name you want for your environment):

cmd (Windows):
```cmd
>mkvirvualenv env
>workon env
```

Bash (macOS and Linux):
```bash
$ python3 -m venv env
$ source env/bin/activate
```

#### Without Database Install

##### 1. Install requirements

install requirements with:
```bash
pip3 install -r requirements_nosql.txt
```
##### 2. Create `neware_parser/config.ini`.

`neware_parser/config.ini` should contain the following (feel free to modify the values):

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

##### 3. Download a dataset file and put it in the appropriate folder.


#### With Database Install

##### 1. Install requirements

install requirements with:
```bash
pip3 install -r requirements.txt
```

##### 2. [Install PostgreSQL](https://www.2ndquadrant.com/en/blog/pginstaller-install-postgresql/).

**Make sure the installation includes the PostgreSQL Unicode ODBC driver** (e.g. ODBC 64-bit).
After the PostgreSQL installation, there is a separate process where you can choose a driver.

Follow the installation instructions to create new user and password. **Remember these for later**.

##### 3. Add the bin path of the install to the Path variable.

##### 4. Run

```bash
psql -U postgres
```

and enter the password you created in Step 3.

##### 5. Enter the following 3 commands in the terminal.

Note: `my_project`, `my_user`, and `my_password` can be changed to your own secret values.

```sql
CREATE DATABASE my_project;

CREATE USER my_user WITH PASSWORD ‘my_password’;

GRANT ALL PRIVILEGES ON DATABASE my_project TO my_user;
```


##### 6. Create `neware_parser/config.ini`.

Note:  `my_project`, `my_user`, and `my_password` can be changed to your own values, but they must be the same as those in Step 6. `your_very_secret_key` should be a very secret key if you care about data security.

`config.ini` should contain the following:

```
[DEFAULT]
Database = my_project
User = my_user
Password = my_password
Host = localhost
Port = 5432
Backend = postgresql
SecretKey = your_very_secret_key
```

##### 7. Download a dataset file and put it in the appropriate folder.


## Using the Software

First, load the virtual environment containing the software in a new terminal. Replace `env` with the name you used when creating the virtual environment.

cmd (Windows):
```cmd
>workon env
```

Bash (macOS and Linux):
```bash
$ source env/bin/activate
```

**If you do not remember the name of your virtual environment**, you can list existing environments with:

```cmd
>workon
```


To quickly see the web page and start developing, run
```bash
python3 manage.py runserver 0.0.0.0:8000
```
then visit `http://localhost:8000/` with a web browser.

When running the code in production, run
```bash
python3 manage.py process_tasks
```
in a separate terminal to allow background tasks (such as parsing data files). This will process the tasks as they are defined.

### ML Smoothing
cmd (Windows)
```cmd
>ml_smoothing.bat path-figures
````

Bash (macOS and Linux)
```Bash
$ sh ml_smoothing.sh path-figures dataset-version "optional-note-to-self"
```

## Theoretical Physics and Computer Science Behind the Software

We hypothesize that we can make [good generalizations](https://github.com/Samuel-Buteau/universal-battery-database/wiki/Generalization-Criteria) by [approximating](https://github.com/Samuel-Buteau/universal-battery-database/wiki/The-Universal-Approximation-Theorem) the functions that map one degradation mechanism to another using neural networks. 

We aim to develop a theory of lithium-ion cells. We first break down the machine learning problem into smaller sub-problems. From there, we develop frameworks to convert the theory to practical implementations. Finally, we apply the method to experimental data and evaluate the result.
