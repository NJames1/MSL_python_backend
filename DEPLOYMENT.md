# Backend Deployment Guide

This guide explains how to deploy the MSL Python Backend to production.

## Quick Deploy to Railway (Recommended)

Railway is the easiest option for deploying this backend. It handles Docker containers, manages PostgreSQL, and provides excellent free tier.

### Prerequisites
- GitHub account (code must be in a GitHub repository)
- Railway account (free: https://railway.app/)

### Steps

1. **Connect GitHub Repository to Railway**
   - Go to https://railway.app/dashboard
   - Click "Create New Project"
   - Select "GitHub Repo"
   - Authorize Railway to access your GitHub account
   - Select `MSL_python_backend` repository

2. **Configure PostgreSQL Database**
   - In Railway dashboard, click "Add Service"
   - Select "PostgreSQL"
   - Railway will automatically create a database
   - The `DATABASE_URL` environment variable will be set automatically

3. **Set Environment Variables**
   - In Railway project settings, go to "Variables"
   - Add the following variables:
     ```
     ENVIRONMENT=production
     DEBUG=False
     ALLOWED_ORIGINS=https://your-frontend-url.vercel.app,https://your-domain.com
     ```

4. **Deploy**
   - Railway automatically deploys when you push to GitHub
   - Check the "Deployments" tab to monitor the build
   - Once deployed, you'll get a public URL like: `https://your-project-xxxxx.up.railway.app`

5. **Update Frontend**
   - Update `MSL_frontend` to use the backend URL:
     - In Vercel dashboard, go to "Settings" → "Environment Variables"
     - Set `VITE_API_BASE_URL=https://your-project-xxxxx.up.railway.app`
   - Redeploy frontend

### Verifying the Deployment

Once deployed, test the backend:

```bash
# Test health endpoint
curl https://your-project-xxxxx.up.railway.app/health

# Test stats endpoint
curl https://your-project-xxxxx.up.railway.app/stats

# View API docs
https://your-project-xxxxx.up.railway.app/docs
```

---

## Alternative: Deploy to Render.com

Render.com is another easy option similar to Railway.

### Steps

1. Push your code to GitHub
2. Go to https://dashboard.render.com
3. Click "New +" → "Web Service"
4. Connect GitHub repository
5. Set:
   - **Build Command**: Leave empty (Render detects Dockerfile)
   - **Start Command**: Leave empty
   - Add PostgreSQL database service
6. Set environment variables in the Render dashboard
7. Deploy

---

## Alternative: Deploy to Heroku (Requires Credit Card)

Heroku now requires a credit card for deployments.

### Steps

1. Install Heroku CLI: `https://devcenter.heroku.com/articles/heroku-cli`
2. Create app: `heroku create your-app-name`
3. Add PostgreSQL: `heroku addons:create heroku-postgresql:hobby-dev`
4. Set variables: `heroku config:set ENVIRONMENT=production`
5. Deploy: `git push heroku main`

---

## Manual Docker Deployment

You can also deploy using Docker manually on any cloud provider.

### Build and push Docker image:

```bash
# Build image
docker build -t your-registry/msl-backend:latest .

# Push to registry (Docker Hub, ECR, etc.)
docker push your-registry/msl-backend:latest

# Run container
docker run \
  -e DATABASE_URL=postgresql://... \
  -e PORT=8000 \
  -p 8000:8000 \
  your-registry/msl-backend:latest
```

---

## Environment Variables Reference

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `DATABASE_URL` | ✓ | - | PostgreSQL connection string |
| `ENVIRONMENT` | | `development` | Set to `production` for deployed apps |
| `DEBUG` | | `True` | Set to `False` in production |
| `ALLOWED_ORIGINS` | | `*` | CORS allowed origins (comma-separated) |
| `PORT` | | `8000` | Server port |
| `HOST` | | `0.0.0.0` | Server host |

---

## Troubleshooting

### Database Connection Errors
- Ensure `DATABASE_URL` is correct
- Check that database service is running
- Verify network connectivity from app to database

### 502 Bad Gateway
- Check application logs in deployment platform
- Ensure app is starting correctly
- Verify environment variables are set

### API Returns 404
- Verify backend is running and accessible
- Check CORS settings
- Ensure frontend is using correct API base URL

### migrations/apply_migrations.py fails
- This is normal - the file is optional
- Backend uses SQLAlchemy to create tables on startup
- The `|| true` ensures deployment continues even if it fails

---

## After Deployment

1. Update frontend's `VITE_API_BASE_URL` environment variable to point to deployed backend
2. Redeploy frontend on Vercel
3. Test the full application at your deployed frontend URL
