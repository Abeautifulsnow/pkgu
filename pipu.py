import argparse
import asyncio
import inspect
import os
import pathlib
import signal
import subprocess
import time
import traceback
from functools import lru_cache
from typing import Union, AnyStr, List, Optional, Tuple, Callable

import orjson
from colorama import init, Fore, Style
from halo import Halo
from loguru import logger
from prettytable import PrettyTable
from pydantic import BaseModel

# 变量赋值
ENV = os.environ.copy()
ENV["PYTHONUNBUFFERED"] = "1"

# 初始化
loggerIns = logger
init()
spinner = Halo(spinner="bouncingBall", interval=100, text_color="cyan")
spinner.text = "checking for updates..."


def import_module(module_name: str) -> None:
    try:
        __import__(module_name)
    except ModuleNotFoundError:
        subprocess.call(["python3", "-m", "pip", "install", "-U", "pip"])

        run_result = subprocess.run(
            ["python3", "-m", "pip", "install", f"{module_name}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        if run_result.returncode != 0:
            run_result.stderr += run_result.stdout
            loggerIns.error(
                f'Install module error: => {run_result.stderr.decode("utf-8")}'
            )
            os.kill(os.getpid(), signal.SIGABRT)


def run_subprocess_cmd(commands: Union[str, list]) -> Tuple[str, bool]:
    src_file_name = pathlib.Path(inspect.getfile(inspect.currentframe())).name
    cmd_str = ""

    if isinstance(commands, str):
        cmd_str = commands
    elif isinstance(commands, list):
        for element in commands:
            if isinstance(element, list):
                loggerIns.error("Error: the element in Commands must be string type.")
                exit(1)

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
            loggerIns.error(
                f"Error: Return Code: {complete_result.returncode}, {err_msg}"
            )
            return stderr.decode("utf-8"), False

    except subprocess.CalledProcessError:
        func_name = inspect.getframeinfo(inspect.currentframe()).function
        loggerIns.error(f"[{src_file_name}] exception in {func_name}")
        complete_result.kill()

        while complete_result.poll() is None:
            loggerIns.info(f"[{src_file_name}] is waiting the child exit.")

        exit(1)


class PackageInfoBase(BaseModel):
    name: AnyStr
    version: AnyStr
    latest_version: AnyStr
    latest_filetype: AnyStr


class AllPackagesExpiredBaseModel(BaseModel):
    packages: List[PackageInfoBase]


class WriteDataToModel(PrettyTable):
    command = "pip list --outdated --format=json"
    spinner.start()

    def __init__(self):
        super().__init__(
            field_names=["Name", "Version", "Latest Version", "Latest FileType"],
            border=True,
        )
        self.ori_data = run_subprocess_cmd(self.command)
        self.model: Optional[AllPackagesExpiredBaseModel] = None
        self.to_model()
        self.packages: Optional[List[List[str]]] = None
        self.success_install: List[str] = []
        self.fail_install: List[str] = []

    def data_to_json(self):
        return orjson.loads(self.ori_data[0])

    @lru_cache(maxsize=1024)
    def to_model(self):
        json = self.data_to_json()
        self.model = AllPackagesExpiredBaseModel(packages=[*json])

    def _get_packages(self):
        return [
            [
                package_info.name.decode(),
                package_info.version.decode(),
                package_info.latest_version.decode(),
                package_info.latest_filetype.decode(),
            ]
            for package_info in self.model.packages
        ]

    def pretty_table(self):
        if self.model:
            spinner.stop()
            self.packages = self._get_packages()
            self.add_rows(self.packages)

        pretty_output = self.get_string()
        if len(self.model.packages) != 0:
            print(pretty_output)
        else:
            awesome = Fore.GREEN + "✔ Awesome!" + Style.RESET_ALL
            print(f"{awesome} All of your dependencies are up-to-date.")

    # TODO: 将罗列出的需要升级的包支持异步async更新
    def _upgrade_packages(self):
        for package_list in self.packages:
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
        spinner.start()
        spinner.text_color = "green"
        spinner.succeed(
            "Successfully installed {} packages. 「{}」".format(
                len(self.success_install), ", ".join(self.success_install)
            )
        )
        spinner.fail(
            "Unsuccessfully installed {} packages. 「{}」".format(
                len(self.fail_install), ", ".join(self.fail_install)
            )
        )

    def statistic_result(self):
        return self._has_packages(self.packages, self._statistic_result)

    def _has_packages(self, /, packages: Optional[List[List[str]]], cb_func: Callable):
        if packages:
            cb_func()

    def __call__(self, *args, **kwargs):
        self.upgrade_packages()
        self.statistic_result()


def upgrade_expired_package(package_name: str, old_version: str, latest_version: str):
    update_cmd = "pip install --upgrade " + f"{package_name}=={latest_version}"
    spinner.spinner = "dots"
    spinner.text_color = ""
    installing_msg = (
        lambda verb: f"{verb} {package_name}, version: from {old_version} to {latest_version}..."
    )
    spinner.start(installing_msg("installing"))
    update_res, update_res_bool = run_subprocess_cmd(update_cmd)

    if update_res_bool:
        spinner.text_color = "green"
        spinner.succeed(installing_msg("installed"))
        return update_res_bool, package_name
    else:
        spinner.text_color = "red"
        spinner.fail(installing_msg("installed failed"))
        return update_res_bool, package_name


async def run_async(class_name: "WriteDataToModel"):
    expired_packages = class_name.packages
    loop = asyncio.get_event_loop()

    cmd_s = [
        loop.run_in_executor(
            None, upgrade_expired_package, *(package[0], package[1], package[2])
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


if __name__ == "__main__":
    parse = argparse.ArgumentParser(description="Upgrade python lib.")
    parse.add_argument(
        "-a",
        "--async_upgrade",
        help="Update the library asynchronously.",
        action="store_true",
    )

    args = parse.parse_args()

    time_s = time.time()
    wdt = WriteDataToModel()
    wdt.pretty_table()

    if args.async_upgrade:
        asyncio.run(run_async(wdt))
    else:
        wdt()

    print(
        Fore.MAGENTA
        + f"Total time elapsed: {Fore.CYAN}{time.time() - time_s} s."
        + Style.RESET_ALL
    )