# Universal Battery Database

- [Installation](#installation)
  * [Prerequisites](#prerequisites)
  * [Installing Dependencies](#installing-dependencies)
  * [Setup](#setup)
- [Windows 10](#windows-10)
  * [Setup](#setup-1)
- [Using the Software](#using-the-software)
  * [ML Smoothing (Linux and macOS)](#ml-smoothing--linux-and-macos-)
-[Stochiometry](#stochiometry)

<small><i><a href='http://ecotrust-canada.github.io/markdown-toc/'>Table of contents generated with markdown-toc</a></i></small>

## Installation

### Prerequisites

- Python 3
- Pip 3


### Installing Dependencies

Install requirements with
```bash
$ pip3 install -r requirements_nosql.txt
```

### Setup

Create a file called `config.ini` in `neware_parser/`, and put the following content within:

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

## Windows 10

Install dependencies with
```
$ pip -r requirements.txt
```

In order to allow background tasks, 
which are super useful in this case,
run in a separate terminal:
```
$ python3 manage.py process_tasks
```

This will process the tasks as they are defined.


To quickly see the webpage and start developing, run
```
$ python3 manage.py runserver 0.0.0.0:8000
```
and then go to your browser and type http://localhost:8000/ for the webpage.

Then, the more tricky part is to install postgresql and configure it. 

- make sure installation includes the PostgreSQL Unicode ODBC driver 
(after the PostgreSQL installation, there is a separate process where you can choose a driver. I selected ODBC 64-bit)

- make sure you create a user with a password you remember.

- add the bin path of the install to the Path variable.

- run the following:
```bash
$ psql -U postgres
```

(Then, you enter the password that you hopefully still remember!!)
```
CREATE DATABASE myproject;

CREATE USER myuser WITH PASSWORD ‘mypassword’;

GRANT ALL PRIVILEGES ON DATABASE myproject TO myuser;
```


- add a file called config.ini in the root directory, with the following content (feel free to modify):
```
[DEFAULT]
Database = myproject
User = myuser
Password = mypassword
Host = localhost
Port = 5432
```

This is for security purposes.


### Setup

Download a dataset file and put it in the appropriate folder.

## Using the Software

### ML Smoothing (Linux and macOS)

Create a file called `run_smoothing.sh` (which is already in gitignore) that specifies the dataset version and takes in two arguments: output path and notes (optional). Then call `smoothing.sh` with these three arguments. Example `run_ml_smoothing.sh`:
```
# $1 specifies the outputpath for figures and $2 is an optional text for notes
sh smoothing.sh $1 TESTING0 $2
```

Then simply run `sh run_smoothing.sh path-figures optional-note-to-self` in a Bash environment.



## Stochiometry
It is reccomended to always use whole numbers. For instance, instead of 0.33, 0.33, 0.33, simply use 1, 1, 1. If there are some very specific ratios that are too inexact to rationalize, you can try to have sub whole numbers.