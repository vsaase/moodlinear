#FROM kaczmarj/ants:2.3.4
FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-devel
ENV http_proxy http://proxy.klima.ads.local:2080
ENV https_proxy http://proxy.klima.ads.local:2080

# copy files
RUN mkdir /mnt/data
RUN mkdir /mnt/pred
# RUN INSTALL_PKGS="rh-python36 rh-python36-python-devel rh-python36-python-setuptools rh-python36-python-pip nss_wrapper \
#         httpd24 httpd24-httpd-devel httpd24-mod_ssl httpd24-mod_auth_kerb httpd24-mod_ldap \
#         httpd24-mod_session atlas-devel gcc-gfortran libffi-devel libtool-ltdl enchant" && \
#     yum install -y centos-release-scl && \
#     yum -y --setopt=tsflags=nodocs install --enablerepo=centosplus $INSTALL_PKGS && \
#     rpm -V $INSTALL_PKGS && \
#     # Remove centos-logos (httpd dependency) to keep image size smaller.
#     rpm -e --nodeps centos-logos && \
#     yum -y clean all --enablerepo='*'
# RUN scl enable rh-python36 bash
# RUN scl enable rh-python36  
RUN pip install nibabel numpy h5py hdf5plugin scipy
COPY ANTs /workspace/ANTs
COPY brainmni /workspace/brainmni
COPY mni_icbm152_nlin_asym_09c /workspace/mni_icbm152_nlin_asym_09c
COPY *.py /workspace/
COPY *.sh /workspace/
ENV PATH="/workspace/ANTs/bin:${PATH}"
RUN chmod +x /workspace/*.sh