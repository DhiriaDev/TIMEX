#!/usr/bin/env bash

coverage run --source=../timexseries/ -m pytest .
coverage-badge -o ../badges/coverage.svg