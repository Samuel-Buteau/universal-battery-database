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
  * [Installing Without a Database](#installing-without-a-database)
  * [Installing With a Database](#installing-with-a-database)
- [Using the Software](#using-the-software)
  * [ML Smoothing](#ml-smoothing)
- [Theoretical Physics and Computer Science Behind the Software](#theoretical-physics-and-computer-science-behind-the-software)
- [Contributing](#contributing)
  * [Code Conventions](#code-conventions)

## Installation

There are two install options:
1. If you only want to play around with modelling and you have a compiled dataset from somewhere else, you can [install without a database](#installing-without-a-database) (you can always install a database later).
2. If you want to use the database features such as parsing and organising experimental data and metadata, you should [install with a database](#installing-with-a-database).

You should run all the given commands in Command Prompt (Windows) or Terminal (Bash environment on macOS and Linux).

### Prerequisites

- [Python 3](https://www.python.org/downloads/)
- [pip and virtualenv](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/)

### Installing Without a Database

#### 1. [Create and activate a new Python environment](https://github.com/Samuel-Buteau/universal-battery-database/wiki/Creating-and-activating-a-new-Python-environment.).

#### 2. Install requirements.
```
pip3 install -r requirements_nosql.txt
```

#### 3. Create `cycling/config.ini`.

Note: You can pick your own `database`, `user`, `password`, `secret_key`.

`cycling/config.ini` should contain:

```
[DEFAULT]
Database = database
User = user
Password = password
Host = localhost
Port = 0000
Backend = sqlite3
SecretKey = secret_key
```

#### 4. Download your dataset file and put it in the appropriate folder.


### Installing With a Database

#### 1. [Create and activate a new Python environment](https://github.com/Samuel-Buteau/universal-battery-database/wiki/Creating-and-activating-a-new-Python-environment.).

#### 2. Install requirements.
```
pip3 install -r requirements.txt
```

#### 3. [Install PostgreSQL](https://www.2ndquadrant.com/en/blog/pginstaller-install-postgresql/).

There is a separate process to choose a driver after the PostgreSQL installation. **Make sure the installation includes the PostgreSQL Unicode ODBC driver** (e.g. ODBC 64-bit).

Follow the installation instructions to create new user and password. **Remember these for later**.

#### 4. Add the bin path of the install to the Path variable.

#### 5. Connect your user.
```bash
psql -U username
```
where `username` is the one created in Step 3, and enter the password after you hit enter.

#### 6. Create your database.

Note: `database`, `user`, and `password` can be changed to your own values.

```sql
CREATE DATABASE database;

CREATE USER user WITH PASSWORD ‘password’;

GRANT ALL PRIVILEGES ON DATABASE database TO user;
```

#### 7. Create `cycling/config.ini`.

Note:  `database`, `user`, and `password` should be changed to match those in Step 6. Choosing a good `secret_key` is crucial if you care about data security.

`cycling/config.ini` should contain the following:

```
[DEFAULT]
Database = database
User = user
Password = password
Host = localhost
Port = 5432
Backend = postgresql
SecretKey = secret_key
```

## Using the Software

[Load the virtual environment](https://github.com/Samuel-Buteau/universal-battery-database/wiki/Loading-a-Python-environment.) containing the software in a new terminal.

To quickly see the web page and start developing, run
```
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

## Contributing

### Code Conventions

Generally, we follow [Google's Python Style Guide](https://github.com/google/styleguide/blob/gh-pages/pyguide.md).
