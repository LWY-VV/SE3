jupyter lab --ip=0.0.0.0 --port=8888 --allow-root --NotebookApp.password="$(echo arclab6210 | python -c 'from notebook.auth import passwd;print(passwd(input()))')"
