# Render Deployment Guide

1. Push this repository to GitHub.
2. Sign in to [Render](https://render.com) and create a **New Web Service**.
3. Link your GitHub repository. Render will auto-detect `render.yaml`.
4. Set environment variables in the Render Dashboard:
   - `AI_PROVIDER`: `replicate` / `fal` / `mock`
   - `REPLICATE_API_TOKEN`: `your_token_here`
5. Click **Deploy**.