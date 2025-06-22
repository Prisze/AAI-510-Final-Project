FROM public.ecr.aws/lambda/python:3.12

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt -t "${LAMBDA_TASK_ROOT}" \
 && rm -rf /root/.cache/pip

COPY . "${LAMBDA_TASK_ROOT}"

CMD ["lambda_handler.handler"]
