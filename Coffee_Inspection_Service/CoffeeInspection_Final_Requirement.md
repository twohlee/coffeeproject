# CoffeeInspection_requirement.md

## 1. 아나콘다 환경
- conda create -n CoffeeInspection python=3.7

## 2. 모듈 설치 명령어
- conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
- conda install -c conda-forge opencv 
- conda install scikit-learn
- conda install matplotlib
- conda install pandas
- conda install --name CoffeeInspection pylint -y
- pip install git+https://github.com/wookayin/gpustat.git@master : 프롬프트창에서 $ gpustat
- conda install flask

## 3. 설치된 모듈 리스트
#
|Name|Version|Build|Channel|
|:--:|:--:|:--:|:--:|                                       
|ansicon                   |1.89.0                   |pypi_0    |pypi|
|astroid                   |2.4.1                    |py37_0||
|blas                      |1.0                      |mkl||
|blessed                   |1.17.8                   |pypi_0|    pypi|
|blessings                 |1.7                      |pypi_0|    pypi|
|ca-certificates           |2020.1.1                 |     0||
|certifi                   |2020.4.5.1               |py37_0||
|click                     |7.1.2                    |  py_0||
|colorama                  |0.4.3                    |  py_0||
|cudatoolkit               |10.2.89              |h74a9793_1||
|cycler                    |0.10.0                   |py37_0||
|flask                     |1.1.2                    |  py_0||
|freetype                  |2.9.1                |ha9979f8_1||
|gpustat                   |1.0.0.dev0               |pypi_0|    pypi|
|hdf5                      |1.8.20               |hac2f561_1||
|icc_rt                    |2019.0.0             |h0cc432a_1||
|icu                       |58.2                 |ha925a31_3||
|intel-openmp              |2020.1                   |   216||
|isort                     |4.3.21                   |py37_0||
|itsdangerous              |1.1.0                    |py37_0||
|jinja2                    |2.11.2                   |  py_0||
|jinxed                    |1.0.0                    |pypi_0|    pypi|
|joblib                    |0.15.1                   |  py_0||
|jpeg                      |9d                   |he774522_0  |  conda-forge|
|kiwisolver                |1.2.0            |py37h74a9793_0||
|lazy-object-proxy         |1.4.3            |py37he774522_0||
|libblas                   |3.8.0           |8_h8933c1f_netlib    |conda-forge|
|libcblas                  |3.8.0           |8_h8933c1f_netlib    |conda-forge|
|libclang                  |9.0.1           |default_hf44288c_0    |conda-forge|
|liblapack                 |3.8.0           |8_h8933c1f_netlib    |conda-forge
|liblapacke                |3.8.0           |8_h8933c1f_netlib    |conda-forge
|libopencv                 |3.4.2                |h20b85fd_0||
|libpng                    |1.6.37               |h2a8f88b_0||
|libtiff                   |4.1.0                |h56a325e_1||
|libwebp                   |1.0.2                |hfa6e2cd_5    |conda-forge|
|lz4-c                     |1.9.2                |h62dcd97_0||
|m2w64-gcc-libgfortran     |5.3.0                         |6||
|m2w64-gcc-libs            |5.3.0                         |7||
|m2w64-gcc-libs-core       |5.3.0                         |7||
|m2w64-gmp                 |6.1.0                         |2||
|m2w64-libwinpthread-git   |5.0.0.4634.697f757            |   2||
|markupsafe                |1.1.1            |py37he774522_0||
|matplotlib                |3.1.3            |        py37_0||
|matplotlib-base           |3.1.3            |py37h64f37c6_0||
|mccabe                    |0.6.1             |       py37_1||
|mkl                       |2020.1             |         216||
|mkl-service               |2.3.0            |py37hb782905_0||
|mkl_fft                   |1.0.15           |py37h14836fe_0||
|mkl_random                |1.1.1            |py37h47e9c7a_0||
|msys2-conda-epoch         |20160418          |            1||
|ninja                     |1.9.0            |py37h74a9793_0||
|numpy                     |1.18.1           |py37h93ca92e_0||
|numpy-base                |1.18.1           |py37hc3f5095_1||
|nvidia-ml-py3             |7.352.0           |       pypi_0|    pypi|
|olefile                   |0.46               |      py37_0||
|opencv                    |3.4.2            |py37h40b0b35_0||
|openssl                   |1.1.1g            |   he774522_0||
|pandas                    |1.0.3            |py37h47e9c7a_0||
|pillow                    |7.1.2            |py37hcc1f983_0||
|pip                       |20.0.2            |       py37_3||
|psutil                    |5.7.0              |      pypi_0 |   pypi|
|py-opencv                 |3.4.2            |py37hc319ecb_0||
|pylint                    |2.5.2             |       py37_0||
|pymongo                   |3.9.0            |py37ha925a31_0||
|pyparsing                 |2.4.7             |         py_0||
|pyqt                      |5.9.2            |py37h6538335_2||
|python                    |3.7.7             |   h81c818b_4||
|python-dateutil           |2.8.1               |       py_0||
|python_abi                |3.7                |     1_cp37m |   conda-forge|
|pytorch                   |1.5.0           |py3.7_cuda102_cudnn7_0    |pytorch|
|pytz                      |2020.1           |          py_0||
|qt                        |5.9.7            |vc14h73c81de_0||
|scikit-learn              |0.22.1           |py37h6288b17_0||
|scipy                     |1.4.1            |py37h9439919_0||
|setuptools                |47.1.1            |       py37_0||
|sip                       |4.19.8           |py37h6538335_0||
|six                       |1.15.0            |         py_0||
|sqlite                    |3.31.1             |  h2a8f88b_1||
|tk                        |8.6.8               | hfa6e2cd_0||
|toml                      |0.10.0          | py37h28b3542_0||
|torchvision               |0.6.0            |    py37_cu102  |  pytorch|
|tornado                   |6.0.4            |py37he774522_1||
|typed-ast                 |1.4.1            |py37he774522_0||
|vc                        |14.1              |   h0510ff6_4||
|vs2015_runtime            |14.16.27012        |  hf0eaf9b_2||
|wcwidth                   |0.2.4               |     pypi_0  |  pypi|
|werkzeug                  |1.0.1                |      py_0||
|wheel                     |0.34.2                |   py37_0||
|wincertstore              |0.2                    |  py37_0||
|wrapt                     |1.11.2           |py37he774522_0||
|xz                        |5.2.5             |   h62dcd97_0||
|zlib                      |1.2.11             |  h62dcd97_4||
|zstd                      |1.4.4              |  ha9fde0e_3||


## 4. MongoDB 설치
- https://www.mongodb.com/download-center/community
- Version : 4.2.7
- OS : Windows x64
- 설치 : default값으로 설치.
- 환경변수 추가 : C:\Program Files\MongoDB\Server\4.2\bin
- 디렉토리 만들어 주기 : C:\data\db
- terminal창 실행
    - 외부접속 허용 : mongod --bind_ip 0.0.0.0
