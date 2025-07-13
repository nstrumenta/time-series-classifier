FROM nstrumenta/developer:3.1.37

COPY requirements.txt .

RUN pip install -r requirements.txt
