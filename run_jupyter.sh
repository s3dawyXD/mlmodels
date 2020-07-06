#!/bin/bash
# Copyright (c) Jupyter Development Team.
# Distributed under the terms of the Modified BSD License.

set -e



# jupyter lab --NotebookApp.password="$(echo hello | python -c 'from notebook.auth import passwd;print(passwd("123"))')"

jupyter lab



