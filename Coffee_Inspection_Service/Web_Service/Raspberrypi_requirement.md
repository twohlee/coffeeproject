# 라즈베리파이 환경

## 1. 라즈베리파이 카메라 모듈
    - Product Name : Pi NoIR - Raspberry Pi Infrared Camera Module
    - Resolution : 8-megapixel
    - Still picture resolution : 3280 x 2464
    - Max image transfer rate
    - 1080p : 30fps(encode and decode)
    - 720p : 60fps

## 2. 라즈베리파이 OS 설치
- SD카드 포멧
- 컴퓨터 관리 -> 저장소 -> 디스크 관리 -> 디스크 1 우클릭 -> 새단순볼륨 생성 -> 완전 포멧 
- https://www.raspberrypi.org/downloads/ 접속
- Raspberry Pi Imager for Windows 실행
- 가이드 대로 수행.

## 3. 라즈베리파이 samba 설치
- sudo apt-get update
- sudo apt-get install samba samba-common-bin
- sudo smbpasswd -a pi
- sudo nano /etc/samba/smb.conf
        [pi]
        comment = pi sharing folder 
        path = /home/pi(공유폴더 경로)
        valid user = pi
        writable = yes
        read only = no
        browseable = yes
- sudo /etc/init.d/samba restart
- 윈도우에서 \\192.168.0.14\pi 라고 입력


## 4. 라즈베리파이 설정
### 4.1 라즈베리파이 원격 접속 설정
    - 메뉴 -> 기본 설정 -> Raspberry Pi Configuration 실행
    - Change Password(선택)
    - sudo apt-get purge realvnc-vnc-server : 충돌을 위해 먼저 제거
    - sudo apt-geet install tightvncserver
    - sudo apt-get install xrdp
### 4.2 라즈베리파이 공유폴더 네트워크 설정
    - 

## 5. 원격 윈도우 설정
    - 메뉴 -> 윈도우 보조프로그램 -> 원격 데스크톱
    - 라즈베리파이 IP 주소 입력
    - username : pi
    - password : 위에서 입력한 패스워드

## 6. 파이썬 환경
    - python 3.7.3

## 7. pip list

|Package|Version|
|:--:|:--:|
|asn1crypto        |0.24.0     
|astroid           |2.1.0      
|asttokens         |1.1.13     
|automationhat     |0.2.0      
|beautifulsoup4    |4.7.1      
|blinker           |1.4        
|blinkt            |0.1.2      
|buttonshim        |0.0.2      
|Cap1xxx           |0.1.3      
|certifi           |2018.8.24  
|chardet           |3.0.4      
|Click             |7.0        
|colorama          |0.3.7      
|colorzero         |1.1        
|cookies           |2.2.1      
|cryptography      |2.6.1      
|docutils          |0.14       
|drumhat           |0.1.0      
|entrypoints       |0.3        
|envirophat        |1.0.0      
|ExplorerHAT       |0.4.2      
|Flask             |1.0.2      
|fourletterphat    |0.1.0      
|gpiozero          |1.5.1      
|html5lib          |1.0.1      
|idna              |2.6        
|isort             |4.3.4      
|itsdangerous      |0.24       
|jedi              |0.13.2     
|Jinja2            |2.10       
|keyring           |17.1.1     
|keyrings.alt      |3.1.1      
|lazy-object-proxy |1.3.1      
|logilab-common    |1.4.2      
|lxml              |4.3.2      
|MarkupSafe        |1.1.0      
|mccabe            |0.6.1      
|microdotphat      |0.2.1      
|mote              |0.0.4      
|motephat          |0.0.2      
|mypy              |0.670      
|mypy-extensions   |0.4.1      
|numpy             |1.16.2     
|oauthlib          |2.1.0      
|olefile           |0.46       
|pantilthat        |0.0.7      
|parso             |0.3.1      
|pgzero            |1.2        
|phatbeat          |0.1.1      
|pianohat          |0.1.0      
|picamera          |1.13       
|piglow            |1.2.5      
|pigpio            |1.44       
|Pillow            |5.4.1      
|pip               |18.1       
|psutil            |5.5.1      
|pycrypto          |2.6.1      
|pygame            |1.9.4.post1
|Pygments          |2.3.1      
|PyGObject         |3.30.4     
|pyinotify         |0.9.6      
|PyJWT             |1.7.0      
|pylint            |2.2.2      
|pymongo           |3.10.1     
|pyOpenSSL         |19.0.0     
|pyserial          |3.4        
|python-apt        |1.8.4.1    
|pyxdg             |0.25       
|rainbowhat        |0.1.0      
|requests          |2.21.0     
|requests-oauthlib |1.0.0      
|responses         |0.9.0      
|roman             |2.0.0      
|RPi.GPIO          |0.7.0      
|RTIMULib          |7.2.1      
|scrollphat        |0.0.7      
|scrollphathd      |1.2.1      
|SecretStorage     |2.3.1      
|Send2Trash        |1.5.0      
|sense-hat         |2.2.0      
|setuptools        |40.8.0     
|simplejson        |3.16.0     
|six               |1.12.0     
|skywriter         |0.0.7      
|sn3218            |1.2.7      
|soupsieve         |1.8        
|spidev            |3.4        
|ssh-import-id     |5.7        
|thonny            |3.2.6      
|touchphat         |0.0.1      
|twython           |3.7.0      
|typed-ast         |1.3.1      
|unicornhathd      |0.0.4      
|urllib3           |1.24.1     
|webencodings      |0.5.1      
|Werkzeug          |0.14.1     
|wheel             |0.32.3     
|wrapt             |1.10.11 