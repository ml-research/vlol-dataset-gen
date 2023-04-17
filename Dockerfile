# Select the base image
#FROM nvcr.io/nvidia/pytorch:21.12-py3
#FROM python:3.9.7-slim
#FROM nvidia/cuda:11.4.0-runtime-ubuntu20.04
FROM nvidia/cuda:11.4.3-devel-ubuntu20.04
#FROM nvidia/cuda:11.6.2-runtime-ubuntu20.04
RUN apt-get update && DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get -y install tzdata
# installations for python
RUN apt-get update && apt-get install -y build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev wget libbz2-dev liblzma-dev
# installations for blender and other
RUN apt-get update && apt-get install -y git subversion cmake libx11-dev libxxf86vm-dev libxcursor-dev libxi-dev libxrandr-dev libxinerama-dev libglew-dev libwayland-dev wayland-protocols libegl-dev libxkbcommon-dev libdbus-1-dev linux-libc-dev sudo
# install prolog dependencies
RUN apt-get update && apt-get install -y gringo swi-prolog
RUN apt-get upgrade -y

# install python and required packages
RUN mkdir /home/python
WORKDIR /home/python
RUN wget https://www.python.org/ftp/python/3.10.2/Python-3.10.2.tgz
RUN tar xzf Python-3.10.2.tgz
WORKDIR /home/python/Python-3.10.2
RUN ./configure --enable-optimizations
RUN make install

COPY modules/requirements.txt /home/python/requirements.txt
RUN pip3 install --upgrade pip
RUN pip3 install -r /home/python/requirements.txt
RUN pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
RUN ln -s /usr/local/bin/python3.10 /usr/bin/python & ln -s /usr/local/bin/pip3.10 /usr/bin/pip

# set up blender as a python module
RUN mkdir /home/blender-git
WORKDIR /home/blender-git
RUN git clone -b blender-v3.3-release --single-branch https://github.com/blender/blender.git

RUN mkdir /home/blender-git/lib
WORKDIR /home/blender-git/lib
RUN svn checkout https://svn.blender.org/svnroot/bf-blender/tags/blender-3.3-release/lib/linux_centos7_x86_64

WORKDIR /home/blender-git/blender
RUN make update

# use for installation of blender 2.9
# install dependencies
#RUN /home/blender-git/blender/build_files/build_environment/install_deps.sh

# remove old libpng
#RUN mv /usr/lib/x86_64-linux-gnu/libpng.a /usr/lib/x86_64-linux-gnu/libpng_no_pic.a

# compile libpng.a with -fPIC flag
#RUN mkdir /home/libpng
#WORKDIR /home/libpng
#RUN wget https://download.sourceforge.net/libpng/libpng-1.6.37.tar.gz
#RUN tar xvfz libpng-1.6.37.tar.gz
#WORKDIR /home/libpng/libpng-1.6.37
#RUN ./configure --enable-shared --prefix=/home/libp --with-pic=yes
#RUN sudo make
#RUN mv /home/libpng/libpng.a /usr/lib/x86_64-linux-gnu/libpng.a

# use precompiled blender lib
#RUN wget https://github.com/Argmaster/PyR3/releases/download/bpy-binaries/libpng-with-pic-linux.tar.gz
#RUN tar xvfz libpng-with-pic-linux.tar.gz
#RUN mv /home/libpng/lib/libpng.a /usr/lib/x86_64-linux-gnu/libpng.a

WORKDIR /home/blender-git/blender
RUN rm ./CMakeLists.txt
COPY modules/CMakeLists.txt ./CMakeLists.txt
RUN make
RUN cp -a /home/blender-git/build_linux/bin/. /usr/local/lib/python3.10/site-packages/

EXPOSE 8282
# create workdir
RUN mkdir /home/workdir
WORKDIR /home/workdir


ENV PYTHONPATH "${PYTHONPATH}:./"
ENV NVIDIA_VISIBLE_DEVICES all