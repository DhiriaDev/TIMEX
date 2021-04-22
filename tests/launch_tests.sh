#!/usr/bin/env bash

coverage run --source=../timexseries/ -m pytest .
coverage-badge -f -o ../badges/coverage.svg