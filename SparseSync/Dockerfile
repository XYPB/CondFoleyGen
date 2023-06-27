FROM ubuntu:22.04

# RUN rm /etc/apt/sources.list.d/cuda.list && rm /etc/apt/sources.list.d/nvidia-ml.list

RUN apt-get update
RUN apt-get install -y sudo

RUN adduser --disabled-password --gecos '' ubuntu
RUN adduser ubuntu sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER ubuntu

SHELL ["/bin/bash", "-c"]

RUN sudo apt-get -qq install curl vim git zip libglib2.0-0 libsndfile1 libsm6 libxext6 libxrender-dev

WORKDIR /home/ubuntu/

RUN curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN bash ./Miniconda3-latest-Linux-x86_64.sh -b
ENV PATH="/home/ubuntu/miniconda3/bin:$PATH"
RUN echo ". /home/ubuntu/miniconda3/etc/profile.d/conda.sh" >> ~/.profile
RUN conda init bash
RUN conda config --set auto_activate_base false

RUN echo $'name: sparse_sync\n\
channels:\n\
  - pytorch\n\
  - conda-forge\n\
  - defaults\n\
dependencies:\n\
  - _libgcc_mutex=0.1=main\n\
  - _openmp_mutex=5.1=1_gnu\n\
  - absl-py=1.2.0=pyhd8ed1ab_0\n\
  - aiohttp=3.8.1=py38h0a891b7_1\n\
  - aiosignal=1.2.0=pyhd8ed1ab_0\n\
  - antlr-python-runtime=4.9.3=pyhd8ed1ab_1\n\
  - async-timeout=4.0.2=pyhd8ed1ab_0\n\
  - attrs=22.1.0=pyh71513ae_1\n\
  - blas=1.0=mkl\n\
  - blinker=1.4=py_1\n\
  - brotli=1.0.9=h166bdaf_7\n\
  - brotli-bin=1.0.9=h166bdaf_7\n\
  - brotlipy=0.7.0=py38h27cfd23_1003\n\
  - bzip2=1.0.8=h7b6447c_0\n\
  - c-ares=1.18.1=h7f98852_0\n\
  - ca-certificates=2022.9.24=ha878542_0\n\
  - cachetools=5.0.0=pyhd8ed1ab_0\n\
  - certifi=2022.9.24=pyhd8ed1ab_0\n\
  - cffi=1.15.0=py38hd667e15_1\n\
  - charset-normalizer=2.0.4=pyhd3eb1b0_0\n\
  - click=8.1.3=py38h578d9bd_0\n\
  - colorama=0.4.5=pyhd8ed1ab_0\n\
  - cryptography=37.0.1=py38h9ce1e76_0\n\
  - cudatoolkit=11.3.1=h2bc3f7f_2\n\
  - cycler=0.11.0=pyhd8ed1ab_0\n\
  - dbus=1.13.18=hb2f20db_0\n\
  - docker-pycreds=0.4.0=py_0\n\
  - einops=0.4.1=pyhd8ed1ab_0\n\
  - expat=2.4.8=h27087fc_0\n\
  - ffmpeg=4.3.2=hca11adc_0\n\
  - ffmpeg-python=0.2.0=py_0\n\
  - fontconfig=2.14.0=h8e229c2_0\n\
  - fonttools=4.25.0=pyhd3eb1b0_0\n\
  - freetype=2.11.0=h70c0345_0\n\
  - frozenlist=1.3.0=py38h0a891b7_1\n\
  - future=0.18.2=py38h578d9bd_5\n\
  - giflib=5.2.1=h7b6447c_0\n\
  - gitdb=4.0.9=pyhd8ed1ab_0\n\
  - gitpython=3.1.27=pyhd8ed1ab_0\n\
  - glib=2.69.1=h4ff587b_1\n\
  - gmp=6.2.1=h295c915_3\n\
  - gnutls=3.6.15=he1e5248_0\n\
  - google-auth=2.9.1=pyh6c4a22f_0\n\
  - google-auth-oauthlib=0.4.6=pyhd8ed1ab_0\n\
  - grpcio=1.42.0=py38hce63b2e_0\n\
  - gst-plugins-base=1.14.0=hbbd80ab_1\n\
  - gstreamer=1.14.0=h28cd5cc_2\n\
  - icu=58.2=hf484d3e_1000\n\
  - idna=3.3=pyhd3eb1b0_0\n\
  - importlib-metadata=4.11.4=py38h578d9bd_0\n\
  - intel-openmp=2021.4.0=h06a4308_3561\n\
  - joblib=1.1.0=pyhd8ed1ab_0\n\
  - jpeg=9e=h7f8727e_0\n\
  - kiwisolver=1.4.2=py38h295c915_0\n\
  - lame=3.100=h7b6447c_0\n\
  - lcms2=2.12=h3be6417_0\n\
  - ld_impl_linux-64=2.38=h1181459_1\n\
  - libblas=3.9.0=12_linux64_mkl\n\
  - libbrotlicommon=1.0.9=h166bdaf_7\n\
  - libbrotlidec=1.0.9=h166bdaf_7\n\
  - libbrotlienc=1.0.9=h166bdaf_7\n\
  - libcblas=3.9.0=12_linux64_mkl\n\
  - libffi=3.3=he6710b0_2\n\
  - libgcc-ng=11.2.0=h1234567_1\n\
  - libgfortran-ng=12.1.0=h69a702a_16\n\
  - libgfortran5=12.1.0=hdcd56e2_16\n\
  - libgomp=11.2.0=h1234567_1\n\
  - libiconv=1.16=h7f8727e_2\n\
  - libidn2=2.3.2=h7f8727e_0\n\
  - liblapack=3.9.0=12_linux64_mkl\n\
  - libpng=1.6.37=hbc83047_0\n\
  - libprotobuf=3.15.8=h780b84a_1\n\
  - libstdcxx-ng=11.2.0=h1234567_1\n\
  - libtasn1=4.16.0=h27cfd23_0\n\
  - libtiff=4.2.0=h2818925_1\n\
  - libunistring=0.9.10=h27cfd23_0\n\
  - libuuid=2.32.1=h7f98852_1000\n\
  - libuv=1.40.0=h7b6447c_0\n\
  - libwebp=1.2.2=h55f646e_0\n\
  - libwebp-base=1.2.2=h7f8727e_0\n\
  - libxcb=1.13=h7f98852_1004\n\
  - libxml2=2.9.14=h74e7548_0\n\
  - lz4-c=1.9.3=h295c915_1\n\
  - markdown=3.4.1=pyhd8ed1ab_0\n\
  - markupsafe=2.1.1=py38h0a891b7_1\n\
  - matplotlib=3.5.1=py38h06a4308_1\n\
  - matplotlib-base=3.5.1=py38ha18d171_1\n\
  - mkl=2021.4.0=h06a4308_640\n\
  - mkl-service=2.4.0=py38h7f8727e_0\n\
  - mkl_fft=1.3.1=py38hd3c417c_0\n\
  - mkl_random=1.2.2=py38h51133e4_0\n\
  - multidict=6.0.2=py38h0a891b7_1\n\
  - munkres=1.1.4=pyh9f0ad1d_0\n\
  - ncurses=6.3=h5eee18b_3\n\
  - nettle=3.7.3=hbbd107a_1\n\
  - numpy=1.23.1=py38h6c91a56_0\n\
  - numpy-base=1.23.1=py38ha15fc14_0\n\
  - oauthlib=3.2.0=pyhd8ed1ab_0\n\
  - omegaconf=2.2.2=py38h578d9bd_0\n\
  - openh264=2.1.1=h4ff587b_0\n\
  - openssl=1.1.1l=h7f98852_0\n\
  - packaging=21.3=pyhd8ed1ab_0\n\
  - pathtools=0.1.2=py_1\n\
  - pcre=8.45=h9c3ff4c_0\n\
  - pillow=9.2.0=py38hace64e9_1\n\
  - pip=22.1.2=py38h06a4308_0\n\
  - promise=2.3=py38h578d9bd_6\n\
  - protobuf=3.15.8=py38h709712a_0\n\
  - psutil=5.9.1=py38h0a891b7_0\n\
  - pthread-stubs=0.4=h36c2ea0_1001\n\
  - pyasn1=0.4.8=py_0\n\
  - pyasn1-modules=0.2.7=py_0\n\
  - pycparser=2.21=pyhd3eb1b0_0\n\
  - pyjwt=2.4.0=pyhd8ed1ab_0\n\
  - pyopenssl=22.0.0=pyhd3eb1b0_0\n\
  - pyparsing=3.0.9=pyhd8ed1ab_0\n\
  - pyqt=5.9.2=py38h05f1152_4\n\
  - pysocks=1.7.1=py38h06a4308_0\n\
  - python=3.8.12=h12debd9_0\n\
  - python-dateutil=2.8.2=pyhd8ed1ab_0\n\
  - python_abi=3.8=2_cp38\n\
  - pytorch=1.11.0=py3.8_cuda11.3_cudnn8.2.0_0\n\
  - pytorch-mutex=1.0=cuda\n\
  - pyu2f=0.1.5=pyhd8ed1ab_0\n\
  - pyyaml=6.0=py38h0a891b7_4\n\
  - qt=5.9.7=h5867ecd_1\n\
  - readline=8.1.2=h7f8727e_1\n\
  - requests=2.28.1=py38h06a4308_0\n\
  - requests-oauthlib=1.3.1=pyhd8ed1ab_0\n\
  - rsa=4.9=pyhd8ed1ab_0\n\
  - scikit-learn=1.0.2=py38h1561384_0\n\
  - scipy=1.8.0=py38h56a6a73_1\n\
  - sentry-sdk=1.9.0=pyhd8ed1ab_0\n\
  - setproctitle=1.2.3=py38h0a891b7_0\n\
  - setuptools=61.2.0=py38h06a4308_0\n\
  - shortuuid=1.0.9=pyha770c72_1\n\
  - sip=4.19.13=py38h295c915_0\n\
  - six=1.16.0=pyhd3eb1b0_1\n\
  - smmap=3.0.5=pyh44b312d_0\n\
  - sqlite=3.39.0=h5082296_0\n\
  - tensorboard=2.9.1=pyhd8ed1ab_0\n\
  - tensorboard-data-server=0.6.0=py38h3e25421_1\n\
  - tensorboard-plugin-wit=1.8.1=pyhd8ed1ab_0\n\
  - threadpoolctl=3.1.0=pyh8a188c0_0\n\
  - tk=8.6.12=h1ccaba5_0\n\
  - torchaudio=0.11.0=py38_cu113\n\
  - torchvision=0.12.0=py38_cu113\n\
  - tornado=6.1=py38h0a891b7_3\n\
  - tqdm=4.64.0=pyhd8ed1ab_0\n\
  - typing-extensions=4.1.1=hd3eb1b0_0\n\
  - typing_extensions=4.1.1=pyh06a4308_0\n\
  - urllib3=1.26.11=py38h06a4308_0\n\
  - wandb=0.12.21=pyhd8ed1ab_0\n\
  - werkzeug=2.2.1=pyhd8ed1ab_0\n\
  - wheel=0.37.1=pyhd3eb1b0_0\n\
  - x264=1!161.3030=h7f98852_1\n\
  - xorg-libxau=1.0.9=h7f98852_0\n\
  - xorg-libxdmcp=1.1.3=h7f98852_0\n\
  - xz=5.2.5=h7f8727e_1\n\
  - yaml=0.2.5=h7f98852_2\n\
  - yarl=1.7.2=py38h0a891b7_2\n\
  - zipp=3.8.1=pyhd8ed1ab_0\n\
  - zlib=1.2.12=h7f8727e_2\n\
  - zstd=1.5.2=ha4553b6_0\n\
  - pip:\n\
    - av==8.1.0\n\
' >> conda_env.yml

RUN conda env create -f conda_env.yml
RUN conda clean -afy
RUN rm ./Miniconda3-latest-Linux-x86_64.sh

SHELL ["conda", "run", "-n", "sparse_sync", "/bin/bash", "-c"]

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "sparse_sync"]
