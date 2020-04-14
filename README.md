# Universal Battery Database

The Universal Battery Database is an open source software that describes and predicts the cycling behaviour and degradation mechanisms of li-ion cells. It simulates cells with different chemistries, architecture, and operating conditions using neural networks.

The Universal Battery Database was developed at the Jeff Dahn Research Group, in collaboration with Tesla Motors/Energy, at Dalhousie University.

## Table of Contents

- [Installation](#installation)
  * [Prerequisites](#prerequisites)
  * [Installing Dependencies and Configuring Environment](#installing-dependencies-and-configuring-environment)
- [Using the Software](#using-the-software)
  * [Recommendations](#recommendations)
  * [Run Scripts](#run-scripts)
  * [ML Smoothing (Linux and macOS)](#ml-smoothing--linux-and-macos-)
- [Stoichiometry](#stoichiometry)

## Installation

### Prerequisites

- [Python 3](https://www.python.org/downloads/)
- [pip and virtualenv](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/)


### Installing Dependencies and Configuring Environment

#### 1. Create a new virtual environment.

cmd (Windows):
```cmd
py -m venv env
```

Bash (macOS and Linux):
```bash
python3 -m venv env
```

#### 2. Activate the virtual environment

cmd (Windows):
```cmd
.\env\Scripts\activate
```

Bash (macOS and Linux):
```bash
source env/bin/activate
```

#### 3. Install requirements

If you do not have a database, install requirements with:
```bash
pip3 install -r requirements_nosql.txt
```

Otherwise, install requirements with:
```bash
pip3 install -r requirements.txt
```


#### 4. [Install PostgreSQL](https://www.2ndquadrant.com/en/blog/pginstaller-install-postgresql/).

**Make sure the installation includes the PostgreSQL Unicode ODBC driver** (e.g. ODBC 64-bitODBC 64-bit).

Follow the installation instructions and create new user and password.

#### 5. Add the bin path of the install to the Path variable.

#### 6. Run

```bash
$ psql -U postgres
```
followed by

```sql
CREATE DATABASE my_project;

CREATE USER my_user WITH PASSWORD ‘my_password’;

GRANT ALL PRIVILEGES ON DATABASE my_project TO my_user;
```


#### 7. Create `config.ini` in the root directory.

With the following example (feel free to modify the values) content:

```
[DEFAULT]
Database = database
User = user
Password = password
Host = localhost
Port = 5432
```

This is for security purposes.

#### 10. Download a dataset file and put it in the appropriate folder.

#### 11. Create a new file, `neware_parser/config.ini`, and put the following (again, feel free to modify the values) within:

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
$ python3 manage.py runserver 0.0.0.0:8000
```
then visit `http://localhost:8000/` with your web browser.

Users are recommended to run
```bash
$ python3 manage.py process_tasks
```
in a separate terminal to allow background tasks. This will process the tasks as they are defined.

### Run Scripts

### ML Smoothing (Linux and macOS)

Create a file called `run_smoothing.sh` (which is already in gitignore) that specifies the dataset version and takes in two arguments: output path and notes (optional). Then call `smoothing.sh` with these three arguments. Example `run_ml_smoothing.sh`:
```bash
# $1 specifies the outputpath for figures and $2 is an optional text for notes
sh smoothing.sh $1 TESTING0 $2
```

Then simply runs `sh run_smoothing.sh path-figures optional-note-to-self` in a Bash environment.



## Stoichiometry
It is recommended to always use whole numbers. For instance, instead of 0.33, 0.33, 0.33, simply use 1, 1, 1. If there are some very specific ratios that are too inexact to rationalize, you can try to have sub whole numbers.