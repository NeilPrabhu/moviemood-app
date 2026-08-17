# MovieMood

Movie recommendation MVP that infers a user's mood from their Spotify playlist and recommends movies.

- **Streamlit app**: root (`MovieMood.py`) — deployed on Streamlit Cloud
- **FastAPI service**: `api/` — deployed on Google Cloud Run

Originally built for UC Berkeley MIDS W210 by Neta Tartakovsky, Josie Ruggieri, Sumedh Shah, Will Dudek, and Neil Prabhu.

## Deploy

### Streamlit
Streamlit Cloud watches the root `MovieMood.py` and installs `requirements.txt`.

### API (Google Cloud Run)
```
cd api
gcloud builds submit --config cloudbuild.yaml --project moviemood-api-89236
gcloud run deploy moviemood-api \
  --image us-west1-docker.pkg.dev/moviemood-api-89236/moviemood/api:v1 \
  --region us-west1 --allow-unauthenticated \
  --memory 1Gi --cpu 1 --port 8000 \
  --set-env-vars REDIS_URL=<upstash-url> \
  --project moviemood-api-89236
```
