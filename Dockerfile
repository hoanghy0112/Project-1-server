FROM python:3.9-alpine

USER python

RUN mkdir -p /home/python/app && chown -R python:python /home/python/app

WORKDIR /home/python/app

COPY --chown=python:python . .

RUN pip install -r requirements.txt

EXPOSE 80

CMD ["waitress-serve", "--host", "127.0.0.1", "app:app"]