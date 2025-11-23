#!/bin/bash
# Script to create GitHub repository and push code

set -e

REPO_NAME="mlops-project"
DESCRIPTION="MLOps Pipeline with CI/CD"

echo "🚀 Creating GitHub repository: $REPO_NAME"

# Check if authenticated
if ! gh auth status &>/dev/null; then
    echo "❌ Not authenticated with GitHub CLI"
    echo "📝 Please run: gh auth login"
    echo "   Then run this script again, or create the repo manually at: https://github.com/new"
    exit 1
fi

# Create repository
echo "📦 Creating repository on GitHub..."
gh repo create "$REPO_NAME" \
    --public \
    --description "$DESCRIPTION" \
    --source=. \
    --remote=origin \
    --push

echo "✅ Repository created and code pushed!"
echo "🌐 View it at: https://github.com/$(gh api user --jq .login)/$REPO_NAME"

