#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from setuptools import setup, find_packages

setup(
    name="multiqa_utils",
    packages=find_packages(),
    version="1.0.0",
    description="Personal Utils for multiqa projects",
    url="https://github.com/drothermel/multiqa_utils/",
    classifiers=[],
    setup_requires=[
        "setuptools>=18.0",
    ],
    #install_requires=[],
)
