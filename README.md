### python 版本 必须 >= 3.5.4 


### install
>```
>pip  install  -r  requirements.txt
>```
### run
>```
>python3 gunicorn wsgi --bind=0.0.0.0:8080 --access-logfile=-
>```

### access

>```
>curl  http://localhost:8080 
>```