FROM node:22-slim AS admin-build

WORKDIR /admin

COPY apps/admin/package*.json ./
RUN npm ci

COPY apps/admin ./

ARG VITE_DEMO_MODE=false
ARG VITE_API_BASE_URL=
ENV VITE_DEMO_MODE=${VITE_DEMO_MODE} \
    VITE_API_BASE_URL=${VITE_API_BASE_URL}

RUN npm run build


FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

COPY requirements.txt /app/requirements.txt

RUN python -m pip install --upgrade pip \
    && pip install --no-cache-dir -r /app/requirements.txt

COPY apps /app/apps
COPY --from=admin-build /admin/dist /app/apps/admin/dist

COPY artifacts/har/har_rf_ucihar.joblib /app/artifacts/har/har_rf_ucihar.joblib
COPY artifacts/har/runtime_v1.joblib /app/artifacts/har/runtime_v1.joblib
COPY artifacts/fall/fall_meta_phone_negatives_v1 /app/artifacts/fall/fall_meta_phone_negatives_v1
COPY artifacts/fall/fall_meta_phone_negatives_v2 /app/artifacts/fall/fall_meta_phone_negatives_v2
RUN mkdir -p /app/artifacts/feedback /app/artifacts/runtime_sessions

COPY db /app/db
COPY fusion /app/fusion
COPY metrics /app/metrics
COPY models /app/models
COPY pipeline /app/pipeline
COPY scripts /app/scripts
COPY services /app/services
COPY .env.example /app/.env.example
COPY .env.compose.example /app/.env.compose.example

RUN chmod +x /app/scripts/run_api.sh \
    && mkdir -p /app/logs \
    && groupadd --system --gid 10001 app \
    && useradd --system --uid 10001 --gid app --home-dir /app app \
    && chown -R app:app /app

USER app

CMD ["./scripts/run_api.sh"]
