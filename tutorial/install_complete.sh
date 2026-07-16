# assuming Debian or Ubuntu Linux OS (i.e. using apt)
# seems working also with Windows Linux Subsystem (WLS) ?
# but best to:
# 1. install VirtualBox: https://www.virtualbox.org/wiki/Downloads
# 2. download Debian VM file: https://sourceforge.net/projects/osboxes/files/v/vb/14-D-b/12.11.0/64bit.7z/download
# 3. use 7zip to uncompress the VM file ( https://www.7-zip.org/download.html )
# 4. load the VM file in VirtualBox


sudo apt-get update && sudo apt-get install -y wget curl p7zip-full 7zip git build-essential

# install Miniconda
curl -LO "http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh" && \
  bash Miniconda3-latest-Linux-x86_64.sh -p $HOME/miniconda -b && \
  rm Miniconda3-latest-Linux-x86_64.sh

# install base environment
conda update -c conda-forge -y conda && \
  conda create -n insarenv -c conda-forge python=3.11 pip mamba
# this should set us to the env if we are not there (we should..)
source $HOME/miniconda/etc/profile.d/conda.sh
mamba init && \
  mamba activate insarenv
# install jupyterlab
mamba install -c conda-forge jupyterlab -y
# install LiCSBAS
cd $HOME && git clone https://github.com/comet-licsar/LiCSBAS && \
  for pck in $(tail -n +2 LiCSBAS/LiCSBAS_requirements.txt); do mamba install -c conda-forge -y $pck; done
  # mamba install -c conda-forge -y --file LiCSBAS/LiCSBAS_requirements.txt
echo "export PATH=\$PATH:\$HOME/LiCSBAS/bin" >> ~/.bashrc && \
echo "export PYTHONPATH=\$PYTHONPATH:\$HOME/LiCSBAS/LiCSBAS_lib" >> ~/.bashrc
pip install git+https://github.com/comet-licsar/licsar_extra.git # to install reunwrapper
# install snaphu for the reunwrapper
wget https://web.stanford.edu/group/radar/softwareandlinks/sw/snaphu/snaphu-v2.0.5.tar.gz && tar -xzf snaphu-v2.0.5.tar.gz && rm snaphu-v2.0.5.tar.gz && \
  cd snaphu-v2.0.5/src && make -f Makefile && mkdir -p /usr/local/man/man1 && sudo make install && cd
source ~/.bashrc
$HOME/LiCSBAS/postBuild  # this will download and extract things
wget https://github.com/comet-licsar/LiCSBAS/blob/main/licsbas_tutorial.ipynb # this will download ipynb
jupyter lab --browser=default licsbas_tutorial.ipynb >/dev/null 2>/dev/null &  # this should open it in jupyter lab and in browser
# need to check if jupyter has the PATH and PYTHONPATH set up well
#
# and check LiCSBAS install
LiCSBAS_check_install.py
