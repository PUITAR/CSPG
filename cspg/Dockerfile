FROM ann-benchmarks

RUN apt-get install -y python3-pip wget unzip cmake
RUN pip3 install pybind11 numpy

# RUN git clone https://github.com/PUITAR/CSPG.git

RUN wget -O CSPG.zip https://anonymous.4open.science/api/repo/CSPG-D338/zip
RUN unzip CSPG.zip -d CSPG

RUN cd CSPG/ANNS; mkdir -p build; cd build; \
    cmake -DCMAKE_PREFIX_PATH=$(python3 -c "import pybind11; import os; print(os.path.dirname(pybind11.__file__))") ..
RUN cd CSPG/ANNS/build; make install
RUN cp CSPG/ANNS/modules/anns* .

RUN python3 -c "import anns"