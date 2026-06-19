# Use Python 3.12 slim image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all repository files (excluding those in .gitignore)
COPY . .

# Run ranking on the sample candidate dataset
CMD ["python", "rank.py", "--candidates", "./data-and-ai-challange/India_runs_data_and_ai_challenge/sample_candidates.json", "--out", "./sample_submission.csv"]
