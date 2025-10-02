FROM nstrumenta/data-job-runner:3.1.44

COPY requirements.txt .

RUN pip install -r requirements.txt
