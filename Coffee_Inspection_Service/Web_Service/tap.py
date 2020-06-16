
#######################################################################################################################################
# [ 모듈 가져 오기 ]
import pymongo
import gridfs
import io

from picamera import PiCamera
from time import sleep
from datetime import datetime

#######################################################################################################################################
# [ 촬영 환경 변수 ]
shot = 3
today = datetime.today().strftime('%Y%m%d_%H%M%S')
ip_address = 'main_server_ip_adress'
print(today)
print(shot)



#######################################################################################################################################
# [ 촬영 조건 ]
camera = PiCamera()
camera.resolution = (2160,2160)
camera.shutter_speed = 15000
camera.sharpness = 70 # -100 ~ 100
# camera.brightness =  # brightness 0 ~ 100
# camera.contrast = # 0 ~ 100
camera.awb_mode = 'auto'
# awb_mode : off, auto, sunlight, cloudy, shade, tungsten, fluorescent, incandescent, flash, horizon
camera.iso = 50


#######################################################################################################################################
# [ 촬영 및 로그데이터 저장 ]
camera.start_preview()
camera.capture('/home/pi/Desktop/Web_rb/static/logdata/raw_data/'+ str(shot) + '_' + str(today) +'.png')
camera.stop_preview()

#######################################################################################################################################
# [ 로그데이터 데이터베이스 저장 ]
filename = '/home/pi/Desktop/Web_rb/static/logdata/raw_data/' + str(shot) + '_' + str(today) + '.png'
datafile = open(filename, 'rb')
thedata = datafile.read()
conn = pymongo.MongoClient(ip_address, 27017)
db = conn.get_database('raspberrypi')
fs = gridfs.GridFS(db, collection = 'logdata')
stored = fs.put(thedata, filename = str(today) + 'logdata')
conn.close()



