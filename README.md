# Universal Battery Database

We aim to simulate li-ion cells with different chemistries, archtecture, and operating conditions.

- [Installation](#installation)
  * [Prerequisites](#prerequisites)
  * [Installing Dependencies](#installing-dependencies)
  * [Installing and Configuring PostgreSQL](#installing-and-configuring-postgresql)
  * [Setup](#setup)
- [Using the Software](#using-the-software)
  * [Recommendations](#recommendations)
  * [ML Smoothing (Linux and macOS)](#ml-smoothing--linux-and-macos-)
- [Stochiometry](#stochiometry)

## Installation

### Prerequisites

- Python 3
- Pip 3


### Installing Dependencies

Install requirements with
```bash
$ pip3 install -r requirements_nosql.txt
```

### Installing and Configuring PostgreSQL 


1. Install PostgreSQL. **Make sure the installation includes the PostgreSQL Unicode ODBC driver.** (You can choose a driver once installation is finished; I selected ODBC 64-bitODBC 64-bit.)
2. Create a new user and password.
3. Add the bin path of the install to the Path variable.
4. Run

```bash
$ psql -U postgres
```
and enter the password you created in step 2.

```sql
CREATE DATABASE my_project;

CREATE USER my_user WITH PASSWORD ‘my_password’;

GRANT ALL PRIVILEGES ON DATABASE my_project TO my_user;
```


5. Create `config.ini` in the root directory, with the following content:

```
[DEFAULT]
Database = my_project
User = my_user
Password = my_password
Host = localhost
Port = 5432
```

This is for security purposes.


### Setup

Download a dataset file and put it in the appropriate folder.

Create a new file, `neware_parser/config.ini`, and put the following within:

```
[DEFAULT]
Database = d
User = u
Password = p
Host = localhost
Port = 0000
Backend = sqlite3
SecretKey = verysecretkeyhaha
```


## Using the Software

To quickly see the webpage and start developing, run
```bash
$ python3 manage.py runserver 0.0.0.0:8000
```
then visit `http://localhost:8000/` for the webpage.

### Recommendations

Run
```bash
$ python3 manage.py process_tasks
```
in a separate terminal to allow background tasks. This will process the tasks as they are defined.

### ML Smoothing (Linux and macOS)

Create a file called `run_smoothing.sh` (which is already in gitignore) that specifies the dataset version and takes in two arguments: output path and notes (optional). Then call `smoothing.sh` with these three arguments. Example `run_ml_smoothing.sh`:
```
# $1 specifies the outputpath for figures and $2 is an optional text for notes
sh smoothing.sh $1 TESTING0 $2
```

Then simply run `sh run_smoothing.sh path-figures optional-note-to-self` in a Bash environment.



## Stochiometry
It is reccomended to always use whole numbers. For instance, instead of 0.33, 0.33, 0.33, simply use 1, 1, 1. If there are some very specific ratios that are too inexact to rationalize, you can try to have sub whole numbers.