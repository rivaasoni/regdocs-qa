# Packages RegDocs Q&A so it runs the same way on any machine.
#
# Build it:
#   docker build -t regdocs-qa .
#
# Run it, passing your API keys in from your .env file:
#   docker run -p 8501:8501 --env-file .env regdocs-qa
#
# Then open http://localhost:8501

# Match the Python version the project was developed against. "slim" is a
# smaller base image than the default, which keeps the build quick.
FROM python:3.11-slim

WORKDIR /app

# Copy and install the dependencies BEFORE copying the rest of the code.
# Docker caches each step, so as long as requirements.txt has not changed it
# reuses the installed packages instead of downloading them again on every
# build. Copying the code first would throw that cache away on every edit.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Now the application code and the prebuilt index.
COPY *.py ./
COPY chunks.json embeddings.npy ./

# Note what is NOT copied: the .env file holding your API keys, and the source
# PDFs. Keys are passed in at run time with --env-file so they never get baked
# into the image, which would leak them to anyone who has the image.

EXPOSE 8501

# --server.address=0.0.0.0 is essential. Streamlit otherwise listens only on
# localhost INSIDE the container, which means nothing outside can reach it and
# the port mapping appears to do nothing.
CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
