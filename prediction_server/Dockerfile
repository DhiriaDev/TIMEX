#syntax=docker/dockerfile:1

FROM timex_utils

WORKDIR /prediction_server

# In order to get the 'utils' directory, we need to build the image from the parent directory.
# Obviously, we need to properly set the paths

ADD pyproject.toml poetry.lock ./

RUN poetry config virtualenvs.create false \
    && poetry update

ADD . .

WORKDIR /

ENV PYTHONPATH "${PYTHONPATH}:/utils"

CMD python prediction_server/server.py