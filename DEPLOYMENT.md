# 🚀 Deployment Guide for Agentic RAG Policy Analyst

## ⚠️ Important: GitHub Pages Limitation

**GitHub Pages cannot run Python/Streamlit applications!** 

GitHub Pages is a **static hosting service** that only serves HTML, CSS, and JavaScript files. It does not provide a Python runtime environment required for Streamlit applications.

### What's Currently Deployed?

- **GitHub Pages URL**: https://stu-ops.github.io/Policy-Analyst/
- **Content**: A static `index.html` page with instructions on how to run the app
- **Purpose**: Provides information and deployment options for the Streamlit app

## ✅ Deployment Options

### Option 1: Run Locally (Best for Development)

This is the simplest way to run the application on your own machine.

#### Prerequisites
- Python 3.11 or higher
- Git
- Gemini API key ([Get one free here](https://aistudio.google.com/apikey))

#### Steps

```bash
# 1. Clone the repository
git clone https://github.com/Stu-ops/Policy-Analyst.git
cd Policy-Analyst

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the Streamlit app
streamlit run app.py

# 4. Open your browser to http://localhost:8501
# 5. Enter your Gemini API key in the sidebar when prompted
```

### Option 2: Deploy to Streamlit Cloud (Recommended for Production)

Streamlit Cloud offers **free hosting** for Streamlit applications with:
- ✅ Automatic deployment from GitHub
- ✅ Free SSL certificates
- ✅ Easy secret management
- ✅ Automatic rebuilds on git push

#### Steps

1. **Fork or Push to GitHub**
   - Ensure your code is in a GitHub repository
   
2. **Sign up for Streamlit Cloud**
   - Go to [https://share.streamlit.io/](https://share.streamlit.io/)
   - Sign in with your GitHub account
   - Authorize Streamlit Cloud to access your repositories

3. **Create New App**
   - Click "New app" button
   - Select your repository: `Stu-ops/Policy-Analyst`
   - Set branch: `main` (or your preferred branch)
   - Set main file path: `app.py`

4. **Configure Secrets**
   - Click on "Advanced settings"
   - In the "Secrets" section, add:
   ```toml
   GEMINI_API_KEY = "your-api-key-here"
   ```

5. **Deploy**
   - Click "Deploy!"
   - Wait for the app to build and deploy (usually 2-5 minutes)
   - Your app will be available at: `https://share.streamlit.io/stu-ops/policy-analyst/`

6. **Update App URL in index.html (Optional)**
   - Once deployed, update the `index.html` file to include a direct link to your live app

### Option 3: Deploy to Hugging Face Spaces

Hugging Face Spaces provides free hosting for ML applications including Streamlit.

#### Steps

1. **Create a Hugging Face Account**
   - Sign up at [https://huggingface.co/](https://huggingface.co/)

2. **Create a New Space**
   - Go to [https://huggingface.co/spaces](https://huggingface.co/spaces)
   - Click "Create new Space"
   - Choose "Streamlit" as the SDK
   - Name your space (e.g., "policy-analyst")

3. **Upload Your Files**
   - Upload `app.py` and all other Python files
   - Upload `requirements.txt`
   - Upload sample documents folder if needed

4. **Add Secrets**
   - Go to Space Settings → Repository secrets
   - Add `GEMINI_API_KEY` with your API key

5. **App Will Deploy Automatically**
   - URL will be: `https://huggingface.co/spaces/your-username/policy-analyst`

### Option 4: Deploy to Render

Render offers free web service hosting with:
- ✅ Automatic deploys from GitHub
- ✅ Free SSL
- ✅ Custom domains

#### Steps

1. **Create Render Account**
   - Sign up at [https://render.com/](https://render.com/)
   - Connect your GitHub account

2. **Create New Web Service**
   - Click "New +" → "Web Service"
   - Connect your GitHub repository
   - Select the repository: `Policy-Analyst`

3. **Configure Service**
   - **Name**: policy-analyst
   - **Environment**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`

4. **Add Environment Variables**
   - Add `GEMINI_API_KEY` with your API key

5. **Deploy**
   - Click "Create Web Service"
   - Wait for deployment to complete

### Option 5: Deploy to Railway

Railway provides simple deployments with a free tier.

#### Steps

1. **Create Railway Account**
   - Sign up at [https://railway.app/](https://railway.app/)
   - Connect GitHub

2. **Create New Project**
   - Click "New Project"
   - Select "Deploy from GitHub repo"
   - Choose `Policy-Analyst`

3. **Configure**
   - Railway will auto-detect Python
   - Add environment variable: `GEMINI_API_KEY`

4. **Generate Domain**
   - Go to Settings → Generate Domain
   - Your app will be available at the generated URL

## 🔧 Configuration Notes

### Environment Variables

All deployment platforms need the following environment variable:

```bash
GEMINI_API_KEY=your-actual-api-key-here
```

Get your free API key from: [https://aistudio.google.com/apikey](https://aistudio.google.com/apikey)

### Requirements

Ensure your `requirements.txt` includes all necessary dependencies:
- streamlit
- google-genai
- langchain and related packages
- faiss-cpu
- rank-bm25
- document processing libraries (pypdf2, python-docx, openpyxl)

### Port Configuration

Most deployment platforms assign a port automatically. The app is configured to work with:
- Local development: `http://localhost:8501`
- Production: Uses platform-assigned port via `$PORT` environment variable

## 📊 Monitoring and Maintenance

### Streamlit Cloud
- Built-in analytics dashboard
- Automatic updates on git push
- Logs available in the web interface

### Other Platforms
- Check platform-specific monitoring tools
- Set up logging as needed
- Monitor API usage for Gemini API

## 🆘 Troubleshooting

### App Won't Start
- Check that all dependencies are in `requirements.txt`
- Verify Python version is 3.11+
- Check environment variables are set correctly

### API Errors
- Verify GEMINI_API_KEY is valid
- Check API quota limits
- Ensure API key has proper permissions

### Deployment Fails
- Check build logs for specific errors
- Verify all file paths are correct
- Ensure no hardcoded local paths

## 📝 Best Practices

1. **Never commit API keys** to the repository
2. Use **environment variables** for secrets
3. Test locally before deploying
4. Use **version control** for tracking changes
5. Monitor API usage to avoid quota issues
6. Keep dependencies updated but test before deploying

## 🔗 Useful Links

- [Streamlit Documentation](https://docs.streamlit.io/)
- [Streamlit Cloud](https://share.streamlit.io/)
- [Gemini API Documentation](https://ai.google.dev/docs)
- [LangChain Documentation](https://python.langchain.com/)
- [Repository](https://github.com/Stu-ops/Policy-Analyst)

## 💡 Need Help?

If you encounter issues:
1. Check the platform-specific documentation
2. Review application logs
3. Verify all configuration settings
4. Test locally to isolate the issue
5. Check GitHub Issues for similar problems

---

**Remember**: GitHub Pages (the current setup) only shows static HTML. To run the actual Streamlit application, you must use one of the deployment options above!
