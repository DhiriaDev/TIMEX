#syntax=docker/dockerfile:1

FROM python:3.8-slim-buster

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    POETRY_HOME="/opt/poetry" \
    POETRY_NO_INTERACTION=1

# prepend poetry and venv to path
ENV PATH="$POETRY_HOME/bin:$PATH"

RUN apt-get update && \
    apt-get install --no-install-recommends -y \
    # deps for installing poetry
    curl \
    # deps for building python deps
    build-essential && \
    python -m ensurepip --upgrade

WORKDIR $POETRY_HOME 
RUN curl -sSL https://install.python-poetry.org | python -

WORKDIR /timex

ADD poetry.lock .
ADD pyproject.toml .
RUN poetry config virtualenvs.create false && poetry install

ADD . .

CMD [ "gunicorn", "-b 0.0.0.0:5000", "-t 0" ,"app:server"]

