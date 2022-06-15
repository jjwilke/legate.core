#!/usr/bin/env python

# Copyright 2021-2022 NVIDIA Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import argparse
import os
import sys
from distutils.command.build_py import build_py
from distutils.core import setup

from setuptools import find_packages

# We need to know the prefix for the installation
# so we can know where to get the library
parser = argparse.ArgumentParser()
parser.add_argument("--prefix", required=False)
parser.add_argument(
    "--recurse", required=False, default=False, action="store_true"
)
args, _ = parser.parse_known_args()


class my_build_py(build_py):
    def run(self):
        if not self.dry_run:
            self.mkpath(self.build_lib)

            # Read our preprocessed C header and insert it as a string into
            # install_info.py so it's installed as part of the python library

            root_dir = os.path.dirname(os.path.realpath(__file__))
            output_dir = os.path.join(root_dir, "legate", "core")
            libpath = repr(os.path.join(args.prefix, "lib"))
            include_dir = os.path.join(args.prefix, "include", "legate")

            with open(os.path.join(include_dir, "legate_c.h.i")) as f:
                header = f.read()

            with open(os.path.join(output_dir, "install_info.py.in")) as f:
                content = f.read()

            content = content.format(header=repr(header), libpath=libpath)

            with open(os.path.join(output_dir, "install_info.py"), "wb") as f:
                f.write(content.encode("utf-8"))

            # header_src = os.path.join(root_dir, "src", "core", "legate_c.h")
            # header = subprocess.check_output(
            #     [
            #         os.getenv("CC", "gcc"),
            #         "-E",
            #         "-DLEGATE_USE_PYTHON_CFFI",
            #         "-I" + str(include_dir),
            #         "-P",
            #         header_src,
            #     ]
            # ).decode("utf-8")

        build_py.run(self)


# If we haven't been called from install.py then do that first
if args.recurse:
    # Remove the recurse argument from the list
    sys.argv.remove("--recurse")
    setup(
        name="legate.core",
        version="22.03",
        packages=find_packages(
            where=".",
            include=["legate*"],
        ),
        cmdclass={"build_py": my_build_py},
    )
else:
    with open("install.py") as f:
        code = compile(f.read(), "install.py", "exec")
        exec(code)
