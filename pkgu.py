import argparse
import asyncio
import hashlib
import inspect
import os
import pathlib
import pickle
import platform
import shutil
import signal
import sqlite3
import subprocess
import sys
import time
import traceback
from abc import ABCMeta, abstractmethod
from functools import lru_cache
from sqlite3 import Connection, Cursor, OperationalError
from typing import (
    Any,
    AnyStr,
    Callable,
    List,
    NewType,
    Optional,
    Sequence,
    Tuple,
    Union,
)

try:
    import importlib.metadata as importlib_metadata
except ModuleNotFoundError:
    import importlib_metadata

import orjson
from colorama import Fore, Style, init
from halo import Halo
from loguru import logger
from prettytable import PrettyTable
from pydantic import VERSION, BaseModel

# The current platform
SYSTEM = platform.system()

if SYSTEM == "Windows":
    from inquirer import Checkbox as IQ_Checkbox
    from inquirer import List as IQ_List
    from inquirer import prompt as IQ_Prompt
else:
    from simple_term_menu import TerminalMenu


# 变量赋值
ENV = os.environ.copy()
ENV["PYTHONUNBUFFERED"] = "1"
__version__ = importlib_metadata.version("pkgu")

# typing
T_NAME = NewType("T_NAME", str)
T_VERSION = NewType("T_VERSION", str)
T_LATEST_VERSION = NewType("T_LATEST_VERSION", str)
T_LATEST_FILETYPE = NewType("T_LATEST_FILETYPE", str)


def clear_lines(num_lines: int):
    """Make use of the escape sequences to clear several lines on terminal.
    It works in mose common terminals, but there may be some variations or
    limitations across different systems.

    Args:
        num_lines (int): how many lines need to be cleared from bottom to top.
    """
    # Move the cursor up 'num_lines' lines
    sys.stdout.write("\033[{}A".format(num_lines))
    # Clear the lines
    sys.stdout.write("\033[2K" * num_lines)
    # Move the cursor back to the beginning of the first cleared line
    sys.stdout.write("\033[{}G".format(0))
    sys.stdout.flush()


def import_module(module_name: str) -> None:
    try:
        __import__(module_name)
    except ModuleNotFoundError:
        subprocess.call(["python3", "-m", "pip", "install", "-U", "pip"])

        run_result = subprocess.run(
            ["python3", "-m", "pip", "install", f"{module_name}", "--no-cache"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        if run_result.returncode != 0:
            run_result.stderr += run_result.stdout
            logger.error(
                f'Install module error: => {run_result.stderr.decode("utf-8")}'
            )
            os.kill(os.getpid(), signal.SIGABRT)


def run_subprocess_cmd(commands: Union[str, list]) -> Tuple[str, bool]:
    """Run shell command in Popen instance.

    Args:
        commands (Union[str, list]): The commands can be string or list.

    Returns:
        Tuple[str, bool]: If the command is executed successfully,
        then return stdout and True, otherwise return stderr and False.
    """
    src_file_name = pathlib.Path(inspect.getfile(inspect.currentframe())).name
    cmd_str = ""

    if isinstance(commands, str):
        cmd_str = commands
    elif isinstance(commands, list):
        for element in commands:
            if isinstance(element, list):
                logger.error("Error: the element in Commands must be string type.")
                sys.exit(1)

            cmd_str = " ".join(commands)

    complete_result = subprocess.Popen(
        cmd_str,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=ENV,
        start_new_session=True,
    )

    try:
        stdout, stderr = complete_result.communicate()

        if complete_result.returncode == 0:
            return stdout.decode("utf-8"), True
        else:
            err_msg = traceback.format_exc()
            logger.error(f"Error: Return Code: {complete_result.returncode}, {err_msg}")
            return stderr.decode("utf-8"), False

    except subprocess.CalledProcessError:
        func_name = inspect.getframeinfo(inspect.currentframe()).function
        logger.error(f"[{src_file_name}] exception in {func_name}")
        complete_result.kill()

        while complete_result.poll() is None:
            logger.info(f"[{src_file_name}] is waiting the child sys.exit.")

        sys.exit(1)


class PackageInfoBase(BaseModel):
    """The basic package infomation."""

    name: AnyStr
    version: AnyStr
    latest_version: AnyStr
    latest_filetype: AnyStr

    if VERSION.split(".")[0] < "2":
        from pydantic import validator

        @validator("name")
        def name_to_str(cls, v: AnyStr):
            if isinstance(v, str):
                return v

            if isinstance(v, bytes):
                return v.decode()

        @validator("version")
        def version_to_str(cls, v: AnyStr):
            if isinstance(v, str):
                return v

            if isinstance(v, bytes):
                return v.decode()

        @validator("latest_version")
        def latest_version_to_str(cls, v: AnyStr):
            if isinstance(v, str):
                return v

            if isinstance(v, bytes):
                return v.decode()

        @validator("latest_filetype")
        def latest_filetype_to_str(cls, v: AnyStr):
            if isinstance(v, str):
                return v

            if isinstance(v, bytes):
                return v.decode()

    else:
        from pydantic import field_validator

        @field_validator("*")
        @classmethod
        def field_to_str(cls, v: AnyStr):
            if isinstance(v, str):
                return v
            if isinstance(v, bytes):
                return v.decode()


class AllPackagesExpiredBaseModel(BaseModel):
    """The list of packages."""

    packages: List[PackageInfoBase]


class WriteDataToModel(PrettyTable):
    __slots__ = ("db", "command")

    def __init__(
        self,
        spinner: "Halo",
        py_env: str,
        cache_path: str,
        cache_valid_duration: int,
        no_cache: bool,
    ):
        self.command = "pip list --outdated --format=json"
        self.db = DAO(cache_path, cache_valid_duration, no_cache)

        self.spinner = spinner
        self.spinner.start()
        super().__init__(
            field_names=["Name", "Version", "Latest Version", "Latest FileType"],
            border=True,
        )
        self.ori_data = self.db.get_result(
            py_env, run_subprocess_cmd, f"{py_env} -m {self.command}"
        )
        # self.ori_data = run_subprocess_cmd(f"{py_env} -m " + self.command)
        self.model: Optional[AllPackagesExpiredBaseModel] = None
        self.to_model()
        self.packages: Optional[
            List[Tuple[T_NAME, T_VERSION, T_LATEST_VERSION, T_LATEST_FILETYPE]]
        ] = None
        self.success_install: List[str] = []
        self.fail_install: List[str] = []

    def data_to_json(self):
        return orjson.loads(self.ori_data[0])

    @lru_cache(maxsize=1024)
    def to_model(self):
        json = self.data_to_json()
        self.model = AllPackagesExpiredBaseModel(packages=[*json])

    def _get_packages(
        self,
    ) -> List[Tuple[T_NAME, T_VERSION, T_LATEST_VERSION, T_LATEST_FILETYPE]]:
        return [
            (
                package_info.name,
                package_info.version,
                package_info.latest_version,
                package_info.latest_filetype,
            )
            for package_info in self.model.packages
        ]

    def pretty_table(self):
        if self.model:
            self.spinner.stop()
            self.packages = self._get_packages()
            self.add_rows(self.packages)

        pretty_output = self.get_string()
        if len(self.model.packages) != 0:
            print(pretty_output)
        else:
            awesome = Fore.GREEN + "✔ Awesome!" + Style.RESET_ALL
            print(f"{awesome} All of your dependencies are up-to-date.")

    def _upgrade_packages(
        self,
        packages: List[Tuple[T_NAME, T_VERSION, T_LATEST_VERSION, T_LATEST_FILETYPE]],
    ):
        """Upgrade packages with synchronous way.

        Args:
            packages:
                (List[Tuple[T_NAME, T_VERSION, T_LATEST_VERSION, T_LATEST_FILETYPE]])
        """
        for package_list in packages:
            package = package_list
            install_res = upgrade_expired_package(package[0], package[1], package[2])

            if install_res[0]:
                self.success_install.append(install_res[1])
            else:
                self.fail_install.append(install_res[1])

    def upgrade_packages(self):
        return self._has_packages(self.packages, self._upgrade_packages)

    def _statistic_result(self):
        print("-" * 60)
        self.spinner.start()
        self.spinner.text_color = "green"
        self.spinner.succeed(
            "Successfully installed {} packages. 「{}」".format(
                len(self.success_install), ", ".join(self.success_install)
            )
        )
        self.spinner.text_color = "red"
        self.spinner.fail(
            "Unsuccessfully installed {} packages. 「{}」".format(
                len(self.fail_install), ", ".join(self.fail_install)
            )
        )
        self.spinner.stop()

    def statistic_result(self):
        return self._has_packages(None, self._statistic_result)

    def _has_packages(self, /, packages: Optional[List[List[str]]], cb_func: Callable):
        if packages:
            cb_func(packages)
        else:
            cb_func()

    # 更新包到最新版本
    def __call__(self, *args, **kwargs):
        self.upgrade_packages()
        self.statistic_result()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.db.cursor.close()
        self.db.conn.close()


class BaseOptions(metaclass=ABCMeta):
    @abstractmethod
    def base_option_single(
        self, title: str, options: List[str], name: Optional[str] = None
    ) -> str:
        raise NotImplementedError("method not implemented")

    @abstractmethod
    def ifUpgradeModules(self) -> str:
        raise NotImplementedError("method not implemented")

    @abstractmethod
    def ifUpdateAllModules(self) -> str:
        raise NotImplementedError("method not implemented")

    @abstractmethod
    def updateOneOfPackages(
        self,
        packages: List[Tuple[T_NAME, T_VERSION, T_LATEST_VERSION, T_LATEST_FILETYPE]],
        # upgrade_func: Callable[[str, str, str], None],
    ) -> Optional[Tuple[str]]:
        raise NotImplementedError("method not implemented")


class UserOptionsForWindows(BaseOptions):
    global IQ_List, IQ_Checkbox, IQ_Prompt

    def __init__(self):
        self.single = IQ_List
        self.mutiple = IQ_Checkbox

    def base_option_single(
        self, title: str, options: List[str], name: Optional[str] = None
    ) -> str:
        questions = [self.single(name, message=title, choices=options)]
        answers = IQ_Prompt(questions)
        return answers.get(name)

    def ifUpgradeModules(self) -> str:
        name = "single"
        title = "continue with the package update?"
        options = ["yes", "no"]
        return self.base_option_single(title, options, name)

    def ifUpdateAllModules(self) -> str:
        name = "single"
        title = "Update all packages listed above or portion of them?"
        options = ["all", "portion"]
        return self.base_option_single(title, options, name)

    def updateOneOfPackages(
        self,
        packages: List[Tuple[T_NAME, T_VERSION, T_LATEST_VERSION, T_LATEST_FILETYPE]],
        # upgrade_func: Callable[[str, str, str], None],
    ) -> Optional[Tuple[str]]:
        title = "Select one of these packages to update"
        options = [f"{package[0]}@{package[1]}=>{package[2]}" for package in packages]
        _name = "multiple"
        terminal_package_option = [self.mutiple(_name, message=title, choices=options)]

        answers = IQ_Prompt(terminal_package_option)
        return answers.get(_name)


class UserOptions(BaseOptions):
    """
    用户选项类，自定义用户选项
    """

    def __init__(self):
        self.tm = TerminalMenu

    def base_option_single(
        self, title: str, options: List[str], name: Optional[str] = None
    ) -> str:
        terminal_menu = self.tm(options, title=title)
        menu_entry_index = terminal_menu.show()
        return options[menu_entry_index]

    def ifUpgradeModules(self) -> str:
        title = "continue with the package update?"
        options = ["yes", "no"]
        return self.base_option_single(title, options)

    def ifUpdateAllModules(self) -> str:
        title = "Update all packages listed above or portion of them?"
        options = ["all", "portion"]
        return self.base_option_single(title, options)

    def updateOneOfPackages(
        self,
        packages: List[Tuple[T_NAME, T_VERSION, T_LATEST_VERSION, T_LATEST_FILETYPE]],
        # upgrade_func: Callable[[str, str, str], None],
    ) -> Optional[Tuple[str]]:
        title = "Select one of these packages to update"
        options = [f"{package[0]}@{package[1]}=>{package[2]}" for package in packages]

        terminal_package_option = self.tm(
            options,
            title=title,
            multi_select=True,
            show_multi_select_hint=True,
            show_search_hint=True,
        )

        terminal_package_option.show()
        return terminal_package_option.chosen_menu_entries


def extract_substrings_with_split(s):
    # Split the string at the '@' and '=>' symbols
    parts = s.split("@")
    package_name = parts[0].strip()

    versions = parts[1].split("=>")
    version1 = versions[0].strip()
    version2 = versions[1].strip()

    return package_name, version1, version2


def upgrade_expired_package(
    package_name: T_NAME, old_version: T_VERSION, latest_version: T_LATEST_VERSION
) -> Tuple[bool, T_NAME]:
    def installing_msg(verb):
        return (
            f"{verb} {package_name}, version: from {old_version} to {latest_version}..."
        )

    with Halo(
        text=installing_msg("installing"),
        spinner="dots",
    ) as spinner:
        update_cmd = "pip install --upgrade " + f"{package_name}=={latest_version}"
        _, update_res_bool = run_subprocess_cmd(update_cmd)

        if update_res_bool:
            spinner.text_color = "green"
            spinner.succeed(installing_msg("installed"))
        else:
            spinner.text_color = "red"
            spinner.fail(installing_msg("installation failed"))

    return update_res_bool, package_name


async def run_async(
    class_name: "WriteDataToModel", expired_packages: Optional[List] = None
):
    if not expired_packages:
        expired_packages = class_name.packages

    loop = asyncio.get_event_loop()

    # TODO: 这个写法有问题，会报错（RuntimeError: threads can only be started once）
    cmd_s = [
        loop.run_in_executor(
            None,
            upgrade_expired_package,
            *(package[0], package[1], package[2]),
        )
        for package in expired_packages
    ]

    res_list = await asyncio.gather(*cmd_s)

    for result in res_list:
        res_bool, pak_name = result
        if res_bool:
            class_name.success_install.append(pak_name)
        else:
            class_name.fail_install.append(pak_name)

    class_name.statistic_result()


def get_python() -> Optional[str]:
    """Return the path of executable python"""
    py_path = shutil.which("python3") or shutil.which("python")

    if py_path is not None:
        return py_path
    else:
        return None


class DAO:
    def __init__(
        self,
        db_file: str,
        expired_time: int,
        no_cache: bool,
        /,
        table_name: str = "cache",
    ) -> None:
        self.db_file = os.path.expanduser(db_file)
        self.expired_time = expired_time
        self.no_cache = no_cache

        self.create_cache_folder()
        self.conn: Connection = next(self.init_db())
        self.cursor: Cursor = self.conn.cursor()

        self.table_name = table_name
        self.create_table(table_name)

    def create_cache_folder(self):
        if not os.path.exists(self.db_file):
            folder_path = os.path.dirname(self.db_file)
            os.makedirs(folder_path, exist_ok=True)

    def init_db(self) -> Connection:
        try:
            yield sqlite3.connect(self.db_file)
        except (OperationalError, Exception) as e:
            logger.error(f"Failed to connect to sqlite db => {self.db_file}")
            raise e

    def _execute_sql(
        self, sql_stmt: str, parameters: Optional[Union[dict, Sequence]] = None
    ):
        if parameters:
            self.cursor.execute(sql_stmt, parameters)
        else:
            self.cursor.execute(sql_stmt)

    def create_table(self, table_name: str):
        create_table = f"""CREATE TABLE IF NOT EXISTS {table_name} (
            key TEXT PRIMARY KEY,
            value BLOB,
            expiration INTEGER
        )"""
        self._execute_sql(create_table)

    def get_from_cache(self, db_key: str):
        # Check if the result exists in the cache and if it has expired
        current_time = int(time.time())
        self._execute_sql(
            f"SELECT value FROM {self.table_name} WHERE key = ? AND expiration > ?",
            (db_key, current_time),
        )
        result = self.cursor.fetchone()
        if result:
            return pickle.loads(result[0])
        else:
            return None

    def store_in_cache(
        self, db_key: str, value: Any, expiration_time: Union[int, float]
    ):
        # Store the result in the cache with an expiration time
        pickled_value = pickle.dumps(value)
        self._execute_sql(
            f"INSERT OR REPLACE INTO {self.table_name} (key, value, expiration) VALUES (?, ?, ?)",
            (db_key, pickled_value, expiration_time),
        )
        self.conn.commit()

    @staticmethod
    def get_cache_key(key: str) -> str:
        # Generate a unique cache key based on script arguments or inputs
        db_key = hashlib.md5(key.encode()).hexdigest()

        return db_key

    def get_result_with_no_cache(
        self,
        cache_key: str,
        nocache_fn: Callable[[Union[str, list]], Tuple[str, bool]],
        param: Union[str, list],
    ) -> Tuple[str]:
        cost_time_res, bool_r = nocache_fn(param)
        if bool_r:
            expiration_time = int(time.time()) + self.expired_time
            self.store_in_cache(cache_key, cost_time_res, expiration_time)
            logger.debug("Origin result")

            return (cost_time_res,)
        else:
            raise ValueError(f"The result if wrong. Command: {param}")

    def get_result_with_cache(self, cache_key: str):
        cache_res = self.get_from_cache(cache_key)
        return cache_res

    def get_result(
        self,
        key: str,
        nocache_fn: Callable[[Union[str, list]], Tuple[str, bool]],
        param: Union[str, list],
    ) -> Tuple[str]:
        cache_key = self.get_cache_key(key)

        if self.no_cache:
            return self.get_result_with_no_cache(cache_key, nocache_fn, param)
        else:
            cache_res = self.get_result_with_cache(cache_key)
            if cache_res:
                return (cache_res,)
            else:
                return self.get_result_with_no_cache(cache_key, nocache_fn, param)


def print_total_time_elapsed(start_time: float, time_end: Optional[float] = None):
    elapsed: str

    if time_end:
        elapsed = time_end - start_time
    else:
        elapsed = time.time() - start_time

    print(
        Fore.MAGENTA + f"Total time elapsed: {Fore.CYAN}{elapsed} s." + Style.RESET_ALL
    )


def parse_args():
    parse = argparse.ArgumentParser(description="Upgrade python lib.", prog="pkgu")
    parse.add_argument(
        "-a",
        "--async_upgrade",
        help="Update the library asynchronously. Default: %(default)s",
        action="store_true",
    )
    parse.add_argument(
        "-d",
        "--cache_folder",
        help="The cache.db file. Default: %(default)s",
        type=str,
        default="~/.cache/cache.db",
    )
    parse.add_argument(
        "-e",
        "--expire_time",
        help="The expiration time. Default: %(default)s",
        type=int,
        default=43200,
    )
    parse.add_argument(
        "--no-cache",
        dest="no_cache",
        help="Whether to use db cache. Default: %(default)s",
        action="store_true",
    )
    parse.add_argument(
        "-v",
        "--version",
        help="Display %(prog)s version and information",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    return parse.parse_args()


def user_option_cls_factory(sys_name: str):
    """The factory functions"""
    if sys_name == "Windows":
        return UserOptionsForWindows()
    else:
        return UserOptions()


def entry():
    """Main entrance."""
    args = parse_args()

    time_s = time.time()
    time_e = 0

    with Halo(
        spinner="bouncingBall",
        interval=100,
        text_color="cyan",
        text="checking for updates...",
    ) as spinner:
        python_env = get_python()
        if python_env is None:
            logger.error("The python3 environment is invalid.")
            return None

        with WriteDataToModel(
            spinner, python_env, args.cache_folder, int(args.expire_time), args.no_cache
        ) as wdt:
            wdt.pretty_table()

            if len(wdt.model.packages) == 0:
                # 打印耗时总时间
                print_total_time_elapsed(time_s)
                return
            else:
                # Get the current time stamp.
                time_e = time.time()

                uo = user_option_cls_factory(SYSTEM)

                flag = uo.ifUpgradeModules()

                if flag == "yes":
                    all_or_portion = uo.ifUpdateAllModules()
                    match all_or_portion:
                        case "all":
                            if args.async_upgrade:
                                asyncio.run(run_async(wdt))
                                # Get the current time stamp.
                                time_e = time.time()
                            else:
                                wdt()
                        case "portion":
                            select_menus = uo.updateOneOfPackages(wdt.packages)
                            if select_menus:
                                select_menus_update = [
                                    extract_substrings_with_split(package_str)
                                    for package_str in select_menus
                                ]

                                if args.async_upgrade:
                                    asyncio.run(run_async(wdt, select_menus_update))
                                    # Get the current time stamp.
                                    time_e = time.time()
                                else:
                                    wdt._has_packages(
                                        select_menus_update, wdt._upgrade_packages
                                    )
                                    wdt.statistic_result()
                else:
                    pass
                # 打印耗时总时间
                print_total_time_elapsed(time_s, time_e)


def main():
    try:
        init()
        entry()
    except KeyboardInterrupt:
        logger.warning("Exit...")


if __name__ == "__main__":
    main()
