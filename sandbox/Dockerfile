FROM python:3.11.13-slim-bookworm
RUN pip install --no-cache-dir numpy==1.26.4 sympy==1.13.3 scipy==1.14.1
COPY runner.py /runner.py
ENV OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
ENTRYPOINT ["python3", "-I", "/runner.py"]
