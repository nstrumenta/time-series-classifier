FROM nstrumenta/developer:3.1.47

COPY requirements.txt .

RUN pip install -r requirements.txt
