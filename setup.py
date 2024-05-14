# Copyright 2024 Li, Ziyao
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Install script for setuptools."""

from setuptools import find_packages
from setuptools import setup

setup(
    name="fastkan",
    version="0.0.1",
    description="Lightning Fast Implementation of Kolmogorov-Arnold Networks",
    author="Li, Ziyao",
    author_email="leeeezy@pku.edu.cn",
    license="Apache License, Version 2.0",
    url="https://github.com/ZiyaoLi/fast-kan",
    packages=find_packages(
        exclude=["img", "notebooks", "tests", "examples"]
    ),
    install_requires=[
        "numpy",
    ],
)
