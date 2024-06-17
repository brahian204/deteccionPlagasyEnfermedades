
NAMEBD = 'images_labels'
USER =  'root'
PASS = '123456789'
HOST = 'localhost'
PORT = '3306'

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.mysql',
        'NAME': NAMEBD,  # Nombre de tu base de datos MySQL
        'USER': USER,      # Usuario de MySQL
        'PASSWORD': PASS,  # Contraseña de MySQL
        'HOST': HOST,   # Host de MySQL (puede variar según tu configuración)
        'PORT': PORT,        # Puerto de MySQL (por defecto es 3306)
    }
}

engine = f'mysql+mysqlconnector://{USER}:{PASS}@{HOST}/{NAMEBD}'
