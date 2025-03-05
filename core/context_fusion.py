# Configuration settings for the application

# Application constants
APP_NAME = "RAG Application"
VERSION = "1.0.0"

# Environment settings
ENVIRONMENT = "development"  # Options: development, testing, production

# Database settings
DATABASE_URL = "sqlite:///rag_application.db"  # Example for SQLite
VECTOR_DB_URL = "http://localhost:8000"  # Example for vector database

# Logging settings
LOG_LEVEL = "INFO"  # Options: DEBUG, INFO, WARNING, ERROR, CRITICAL

# Other settings
MAX_RETRIES = 3
TIMEOUT = 30  # in seconds

# Add any additional configuration settings as needed