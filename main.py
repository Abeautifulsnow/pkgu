import inspect
import os
import pathlib
import signal
import subprocess
import traceback
from functools import lru_cache
from typing import Union, AnyStr, List, Optional

import orjson
from prettytable import PrettyTable
from pydantic import BaseModel

ENV = os.environ.copy()
ENV['PYTHONUNBUFFERED'] = '1'


def import_module(module_name: str) -> None:
    try:
        __import__(module_name)
    except ModuleNotFoundError:
        subprocess.call(['python3', '-m', 'pip', 'install', '-U', 'pip'])

        run_result = subprocess.run(['python3', '-m', 'pip', 'install', f'{module_name}'], stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE)

        if run_result.returncode != 0:
            run_result.stderr += run_result.stdout
            print(f'Install module error: => {run_result.stderr.decode("utf-8")}')
            os.kill(os.getpid(), signal.SIGABRT)


def run_subprocess_cmd(commands: Union[str, list]):
    src_file_name = pathlib.Path(inspect.getfile(inspect.currentframe())).name
    cmd_str = ''

    if isinstance(commands, str):
        cmd_str = commands
    elif isinstance(commands, list):
        for element in commands:
            if isinstance(element, list):
                print('Error: the element in Commands must be string type.')
                exit(1)

            cmd_str = " ".join(commands)

    complete_result = subprocess.Popen(cmd_str, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=ENV,
                                       start_new_session=True)

    try:
        stdout, stderr = complete_result.communicate()

        if complete_result.returncode == 0:
            return stdout.decode('utf-8')
        else:
            err_msg = traceback.format_exc()
            print(f'Error: Return Code: {complete_result.returncode}, {err_msg}')
            return stderr.decode('utf-8')

    except subprocess.CalledProcessError:
        func_name = inspect.getframeinfo(inspect.currentframe()).function
        print(f'[{src_file_name}] exception in {func_name}')
        complete_result.kill()

        while complete_result.poll() is None:
            print(f'[{src_file_name}] is waiting the child exit.')

        exit(1)


class PackageInfoBase(BaseModel):
    name: AnyStr
    version: AnyStr
    latest_version: AnyStr
    latest_filetype: AnyStr


class AllPackagesExpiredBaseModel(BaseModel):
    packages: List[PackageInfoBase]


class WriteDataToModel(PrettyTable):
    command = 'pip list --outdated --format=json'

    def __init__(self):
        super().__init__(field_names=['Name', 'Version', 'Latest Version', 'Latest FileType'], border=True)
        self.ori_data = run_subprocess_cmd(self.command)
        self.model: Optional[AllPackagesExpiredBaseModel] = None
        self.to_model()

    def data_to_json(self):
        return orjson.loads(self.ori_data)

    @lru_cache(maxsize=1024)
    def to_model(self):
        json = self.data_to_json()
        self.model = AllPackagesExpiredBaseModel(packages=[*json])

    def pretty_table(self):
        if self.model:
            packages = [[package_info.name.decode(), package_info.version.decode(), package_info.latest_version.decode(),
                         package_info.latest_filetype.decode()] for package_info in self.model.packages]
            self.add_rows(packages)
        print(self.get_string())

    # TODO: 将罗列出的需要升级的包全部更新


if __name__ == '__main__':
    wdt = WriteDataToModel()
    wdt.pretty_table()
