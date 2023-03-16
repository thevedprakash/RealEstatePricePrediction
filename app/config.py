from ap import app
from flaskext.mysql import MySQL

mysql = MySQL()

app.config['MYSQL_DATABASE_USER'] = 'asteroidris'
app.config['MYSQL_DATABASE_PASSWORD'] = "NIGam@30"
app.config['MYSQL_DATABASE_DB'] = "usa_house"
app.config['MYSQL_DATABASE_HOST'] = 'localhost'

mysql.init_app(app)