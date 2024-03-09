# pkgu

Find the out-dated packages installed by the Pip tool and update them. Inspired by 👉[depu(Go)](https://github.com/kevwan/depu).
However, `pkgu` supports full and partial updates, which is more convenient and flexible. It also supports cross-platform(Windows, linux, macos). 🤓 To retrieve the data more fast after the first-time(Or don't have to execute command again), I use sqlite as cache db to store the data and can quickly read it from db and present it to the console💻.

👉 However, only python3.10 and above are available now.

## Usage

- Run code from source code

First, you need to install dependencies.

if you don't have `poetry` tool, please install it first. 🔗: [poetry installation](https://python-poetry.org/docs/#installation), otherwise, install packages directly.

```bash
poetry install
```

and then, run `pkgu.py` script.

```bash
python3 pkgu.py
```

- Use it through pip - ***Highly Recommended***

```bash
pip3 install pkgu
```

after the installation is complete, `pkgu` executable file will be written to the python bin directory and you can enter `pkgu -h` command on your terminal to learn how to use it.

```bash
# Usage
usage: pkgu [-h] [-v] {update,remove} ...

Upgrade and uninstall python package.

options:
  -h, --help       show this help message and exit
  -v, --version    Display pkgu version and information

Available commands:
  {update,remove}  Available commands
    update         Update python package.
    remove         remove python package with its dependencies.

# ------ update ------
options:
  -h, --help            show this help message and exit
  -a, --async_upgrade   Update the library asynchronously. Default: False
  -d CACHE_FOLDER, --cache_folder CACHE_FOLDER
                        The cache.db file. Default: ~/.cache/cache.db
  -e EXPIRE_TIME, --expire_time EXPIRE_TIME
                        The expiration time. Default: 43200
  --no-cache            Whether to use db cache. Default: False

# ------ remove ------
options:
  -h, --help    show this help message and exit
  -l, --list    list unused dependencies, but don't uninstall them.
  -L, --leaves  list leaves (packages which are not used by any others).
  -y, --yes     don't ask for confirmation of uninstall deletions.
  -f, --freeze  list leaves (packages which are not used by any others) in requirements.txt format
```

## ScreenShoot

> 1. No packages need to be upgraded.

![img.png](https://raw.githubusercontent.com/Abeautifulsnow/pkgu/main/screenshoot/img.png)

> 2. Upgrade some expired packages.

![img_4.png](https://raw.githubusercontent.com/Abeautifulsnow/pkgu/main/screenshoot/img_4.png)
![img_2.png](https://raw.githubusercontent.com/Abeautifulsnow/pkgu/main/screenshoot/img_2.png)
![img_3.png](https://raw.githubusercontent.com/Abeautifulsnow/pkgu/main/screenshoot/img_3.png)

> 3. Update the pkg synchronously

![img_1.png](https://raw.githubusercontent.com/Abeautifulsnow/pkgu/main/screenshoot/img_1.png)

> 4. Update the pkg asynchronously

![img_5.png](https://raw.githubusercontent.com/Abeautifulsnow/pkgu/main/screenshoot/img_5.png)

We can see that the async method is faster than sync method about 9 seconds(Only in this test situation).
So now it can support to update the python libraries asynchronously. 🥳

> 5. Support for selectable update packages

- List all availbable packages

![img_6.png](https://raw.githubusercontent.com/Abeautifulsnow/pkgu/main/screenshoot/img_6.png)

- Select ths part of package to be updated

![img_7.png](https://raw.githubusercontent.com/Abeautifulsnow/pkgu/main/screenshoot/img_7.png)

> 6. Support to use cache result from sqlite db file.

This improve the expirence that how we list the out-dated packages when they are huge to collect, and then there also is a cli flag `--no-cache` to control whether should to use cache.

> 7. !!!New - Support to uninstall package with its dependencies together.

![img_8.png](https://raw.githubusercontent.com/Abeautifulsnow/pkgu/main/screenshoot/img_8.png)
