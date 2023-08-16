# pkgu

Find the out-dated packages installed by the Pip tool and update them. Inspired by 👉[depu](https://github.com/kevwan/depu)

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

- Use it through pip

```bash
pip3 install pkgu
```

after the installation is complete, you can enter `pkgu -h` command on your terminal to learn how to use it.

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

> 5. !!!New - Support for selectable update packages

* List all availbable packages

![img_6.png](https://raw.githubusercontent.com/Abeautifulsnow/pkgu/main/screenshoot/img_6.png)

* Select ths part of package to be updated

![img_7.png](https://raw.githubusercontent.com/Abeautifulsnow/pkgu/main/screenshoot/img_7.png)