#!/usr/bin/env bash
# =============================================================================
# Tetra KG Agent - GCP Cloud Run Deployment Script
# =============================================================================
# This script deploys the Tetra Knowledge Graph Agent to Google Cloud Run.
#
# Prerequisites:
#   - gcloud CLI installed and authenticated
#   - Docker installed (for local builds only)
#   - GCP project with billing enabled
#
# Usage:
#   ./scripts/deploy-cloudrun.sh [options]
#
# Options:
#   --project PROJECT_ID    GCP project ID (required or set GCP_PROJECT env)
#   --region REGION         GCP region (default: us-central1)
#   --service NAME          Cloud Run service name (default: tetra-kg-agent)
#   --setup-secrets         Create/update secrets interactively in Secret Manager
#   --secrets-from-env      Load secrets from local .env file (non-interactive)
#   --local-build           Build locally instead of Cloud Build
#   --skip-build            Skip build, just deploy existing image
#   --allow-unauthenticated Allow public access (default: requires auth)
#   --dry-run               Show commands without executing
#   --help                  Show this help message
# =============================================================================

set -euo pipefail

# ========================== Configuration ==========================
# Override these via environment variables or command line flags

GCP_PROJECT="${GCP_PROJECT:-}"
GCP_REGION="${GCP_REGION:-us-central1}"
SERVICE_NAME="${SERVICE_NAME:-tetra-kg-agent}"
AR_REPO="${AR_REPO:-tetra-repo}"
IMAGE_NAME="${IMAGE_NAME:-tetra-kg-agent}"

# Cloud Run settings (Gen 2)
MEMORY="${MEMORY:-4Gi}"
CPU="${CPU:-2}"
MIN_INSTANCES="${MIN_INSTANCES:-0}"
MAX_INSTANCES="${MAX_INSTANCES:-10}"
TIMEOUT="${TIMEOUT:-300}"
CONCURRENCY="${CONCURRENCY:-80}"
EXECUTION_ENV="${EXECUTION_ENV:-gen2}"
CPU_BOOST="${CPU_BOOST:-true}"
SESSION_AFFINITY="${SESSION_AFFINITY:-true}"

# Feature flags
SETUP_SECRETS=false
SECRETS_FROM_ENV=false
LOCAL_BUILD=false
SKIP_BUILD=false
ALLOW_UNAUTHENTICATED=false
DRY_RUN=false

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ========================== Helper Functions ==========================

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

run_cmd() {
    if [ "$DRY_RUN" = true ]; then
        echo -e "${YELLOW}[DRY-RUN]${NC} $*"
    else
        log_info "Running: $*"
        "$@"
    fi
}

show_help() {
    head -30 "$0" | tail -25
    exit 0
}

# ========================== Argument Parsing ==========================

while [[ $# -gt 0 ]]; do
    case $1 in
        --project)
            GCP_PROJECT="$2"
            shift 2
            ;;
        --region)
            GCP_REGION="$2"
            shift 2
            ;;
        --service)
            SERVICE_NAME="$2"
            shift 2
            ;;
        --setup-secrets)
            SETUP_SECRETS=true
            shift
            ;;
        --secrets-from-env)
            SECRETS_FROM_ENV=true
            shift
            ;;
        --local-build)
            LOCAL_BUILD=true
            shift
            ;;
        --skip-build)
            SKIP_BUILD=true
            shift
            ;;
        --allow-unauthenticated)
            ALLOW_UNAUTHENTICATED=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --help|-h)
            show_help
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# ========================== Validation ==========================

log_info "Validating prerequisites..."

# Check for required tools
if ! command -v gcloud &> /dev/null; then
    log_error "gcloud CLI is required but not installed."
    log_info "Install: https://cloud.google.com/sdk/docs/install"
    exit 1
fi

if [ "$LOCAL_BUILD" = true ] && ! command -v docker &> /dev/null; then
    log_error "Docker is required for local builds but not installed."
    exit 1
fi

# Validate GCP project
if [ -z "$GCP_PROJECT" ]; then
    GCP_PROJECT=$(gcloud config get-value project 2>/dev/null || true)
    if [ -z "$GCP_PROJECT" ]; then
        log_error "GCP project not set. Use --project or set GCP_PROJECT env variable."
        exit 1
    fi
fi

log_success "Using GCP project: $GCP_PROJECT"
log_info "Region: $GCP_REGION"
log_info "Service name: $SERVICE_NAME"

# Set the project for all gcloud commands
run_cmd gcloud config set project "$GCP_PROJECT"

# Get the script's directory and project root (used throughout)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# ========================== Enable APIs ==========================

log_info "Enabling required GCP APIs..."

APIS=(
    "run.googleapis.com"
    "artifactregistry.googleapis.com"
    "cloudbuild.googleapis.com"
    "secretmanager.googleapis.com"
)

for api in "${APIS[@]}"; do
    run_cmd gcloud services enable "$api" --quiet
done

log_success "APIs enabled"

# ========================== Artifact Registry ==========================

log_info "Setting up Artifact Registry..."

# Check if repository exists
if ! gcloud artifacts repositories describe "$AR_REPO" \
    --location="$GCP_REGION" &>/dev/null; then
    log_info "Creating Artifact Registry repository: $AR_REPO"
    run_cmd gcloud artifacts repositories create "$AR_REPO" \
        --location="$GCP_REGION" \
        --repository-format=docker \
        --description="Tetra KG Agent Docker images"
else
    log_info "Artifact Registry repository '$AR_REPO' already exists"
fi

# Configure Docker authentication
run_cmd gcloud auth configure-docker "${GCP_REGION}-docker.pkg.dev" --quiet

IMAGE_URI="${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT}/${AR_REPO}/${IMAGE_NAME}"

# ========================== Secret Manager ==========================

if [ "$SETUP_SECRETS" = true ]; then
    log_info "Setting up secrets in Secret Manager..."

    # List of secrets needed by the application
    declare -A SECRETS=(
        ["GOOGLE_API_KEY"]="Google Gemini API key"
        ["OPENROUTER_API_KEY"]="OpenRouter API key for Cerebras"
        ["NCBI_API_KEY"]="NCBI API key for PubMed (optional)"
        ["LANGFUSE_PUBLIC_KEY"]="Langfuse public key (optional)"
        ["LANGFUSE_SECRET_KEY"]="Langfuse secret key (optional)"
    )

    for secret_name in "${!SECRETS[@]}"; do
        description="${SECRETS[$secret_name]}"

        # Check if secret exists
        if ! gcloud secrets describe "$secret_name" &>/dev/null; then
            log_info "Creating secret: $secret_name ($description)"

            # Prompt for value
            echo -n "Enter value for $secret_name (or press Enter to skip): "
            read -r secret_value

            if [ -n "$secret_value" ]; then
                echo -n "$secret_value" | run_cmd gcloud secrets create "$secret_name" \
                    --data-file=- \
                    --replication-policy="automatic"
                log_success "Secret '$secret_name' created"
            else
                log_warning "Skipping secret: $secret_name"
            fi
        else
            log_info "Secret '$secret_name' already exists"

            # Offer to update
            echo -n "Update secret $secret_name? (y/N): "
            read -r update_choice
            if [[ "$update_choice" =~ ^[Yy]$ ]]; then
                echo -n "Enter new value for $secret_name: "
                read -r secret_value
                if [ -n "$secret_value" ]; then
                    echo -n "$secret_value" | run_cmd gcloud secrets versions add "$secret_name" \
                        --data-file=-
                    log_success "Secret '$secret_name' updated"
                fi
            fi
        fi
    done

    log_success "Secrets configured"
fi

# Load secrets from .env file (non-interactive)
if [ "$SECRETS_FROM_ENV" = true ]; then
    log_info "Loading secrets from .env file..."

    ENV_FILE="$PROJECT_ROOT/.env"
    if [ ! -f "$ENV_FILE" ]; then
        # Try script directory parent
        ENV_FILE="$(dirname "$SCRIPT_DIR")/.env"
    fi

    if [ -f "$ENV_FILE" ]; then
        log_info "Found .env file: $ENV_FILE"

        # Read each secret from .env and create/update in Secret Manager
        for secret_name in GOOGLE_API_KEY OPENROUTER_API_KEY NCBI_API_KEY LANGFUSE_PUBLIC_KEY LANGFUSE_SECRET_KEY; do
            # Extract value from .env file
            secret_value=$(grep -E "^${secret_name}=" "$ENV_FILE" 2>/dev/null | cut -d'=' -f2- | tr -d '"' | tr -d "'" || true)

            if [ -n "$secret_value" ] && [ "$secret_value" != "" ]; then
                if ! gcloud secrets describe "$secret_name" &>/dev/null; then
                    log_info "Creating secret: $secret_name"
                    echo -n "$secret_value" | run_cmd gcloud secrets create "$secret_name" \
                        --data-file=- \
                        --replication-policy="automatic"
                else
                    log_info "Updating secret: $secret_name"
                    echo -n "$secret_value" | run_cmd gcloud secrets versions add "$secret_name" \
                        --data-file=-
                fi
                log_success "Secret '$secret_name' configured"
            else
                log_warning "Secret '$secret_name' not found in .env or empty, skipping"
            fi
        done

        log_success "Secrets loaded from .env"
    else
        log_error ".env file not found. Create one from .env.example"
        exit 1
    fi
fi

# ========================== Build Image ==========================

cd "$PROJECT_ROOT"

if [ "$SKIP_BUILD" = true ]; then
    log_info "Skipping build (--skip-build specified)"
    log_info "Using existing image: ${IMAGE_URI}:latest"
elif [ "$LOCAL_BUILD" = true ]; then
    # Local Docker build + push
    log_info "Building locally with Docker..."
    run_cmd docker build -t "${IMAGE_URI}:latest" .
    run_cmd docker push "${IMAGE_URI}:latest"
    log_success "Image built and pushed: ${IMAGE_URI}:latest"
else
    # Cloud Build (recommended - faster, no local Docker needed)
    log_info "Building with Cloud Build (this may take 5-10 minutes)..."
    run_cmd gcloud builds submit \
        --tag "${IMAGE_URI}:latest" \
        --timeout=1800s \
        --machine-type=e2-highcpu-8 \
        .
    log_success "Image built and pushed: ${IMAGE_URI}:latest"
fi

# ========================== Deploy to Cloud Run ==========================

log_info "Deploying to Cloud Run..."

# Build the deployment command (Cloud Run Gen 2)
DEPLOY_CMD=(
    gcloud run deploy "$SERVICE_NAME"
    --image "${IMAGE_URI}:latest"
    --region "$GCP_REGION"
    --platform managed
    --execution-environment "$EXECUTION_ENV"
    --memory "$MEMORY"
    --cpu "$CPU"
    --min-instances "$MIN_INSTANCES"
    --max-instances "$MAX_INSTANCES"
    --timeout "$TIMEOUT"
    --concurrency "$CONCURRENCY"
    --port 8080
    --cpu-throttling
)

# Add CPU boost for faster cold starts (Gen 2 feature)
if [ "$CPU_BOOST" = true ]; then
    DEPLOY_CMD+=(--cpu-boost)
fi

# Add session affinity for Streamlit (stateful app needs sticky sessions)
if [ "$SESSION_AFFINITY" = true ]; then
    DEPLOY_CMD+=(--session-affinity)
fi

# Add secrets as environment variables
# Only add secrets that exist
for secret in GOOGLE_API_KEY OPENROUTER_API_KEY NCBI_API_KEY LANGFUSE_PUBLIC_KEY LANGFUSE_SECRET_KEY; do
    if gcloud secrets describe "$secret" &>/dev/null; then
        DEPLOY_CMD+=(--set-secrets "${secret}=${secret}:latest")
    fi
done

# Allow unauthenticated access if specified
if [ "$ALLOW_UNAUTHENTICATED" = true ]; then
    DEPLOY_CMD+=(--allow-unauthenticated)
else
    DEPLOY_CMD+=(--no-allow-unauthenticated)
fi

# Execute deployment
run_cmd "${DEPLOY_CMD[@]}"

log_success "Deployment complete!"

# ========================== Get Service URL ==========================

if [ "$DRY_RUN" = false ]; then
    SERVICE_URL=$(gcloud run services describe "$SERVICE_NAME" \
        --region "$GCP_REGION" \
        --format='value(status.url)')

    echo ""
    echo "=============================================="
    log_success "Tetra KG Agent deployed successfully!"
    echo "=============================================="
    echo ""
    echo "Service URL: $SERVICE_URL"
    echo ""

    if [ "$ALLOW_UNAUTHENTICATED" = false ]; then
        log_info "Note: Service requires authentication."
        log_info "To test locally, use:"
        echo ""
        echo "  curl -H \"Authorization: Bearer \$(gcloud auth print-identity-token)\" $SERVICE_URL"
        echo ""
        log_info "To allow public access, run:"
        echo ""
        echo "  gcloud run services add-iam-policy-binding $SERVICE_NAME \\"
        echo "    --region=$GCP_REGION \\"
        echo "    --member='allUsers' \\"
        echo "    --role='roles/run.invoker'"
    fi

    echo ""
    log_info "To view logs:"
    echo "  gcloud run services logs tail $SERVICE_NAME --region=$GCP_REGION"
    echo ""
    log_info "To delete the service:"
    echo "  gcloud run services delete $SERVICE_NAME --region=$GCP_REGION"
fi
